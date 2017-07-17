"""Storage format for frames from openai gym"""
import argparse
import gzip
import logging
import os
import os.path
import pickle
import time
from multiprocessing.connection import Client

import gym
import numpy as np

import log_coordinator
from humanrl.utils import ACTION_MEANING

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument(
    '-f',
    '--frames-dir',
    type=str,
    default="/tmp/pong/frames",
    help="Directory to read and write frames to")


class Episode(object):
    def __init__(self):
        self.frames = []
        self.info = {}
        self.version = 2
        self.path = None

    def __setstate__(self, state):
        """ This method controls backwards-compatibility for unpickling: it's called every
        time a Frame is unpickled, so we can handle class-format changes easily."""
        if 'version' not in state.keys():
            self.__dict__.update(version=1)
        if 'path' not in state.keys():
            self.__dict__.update(path=None)
        self.__dict__.update(state)


class Frame(object):
    REAL = "real"
    PROPOSED = "proposed"

    def __init__(self,
                 observation=None,
                 action=None,
                 reward=None,
                 image=None,
                 info=None,
                 label=None,
                 prediction=None):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.image = image
        self.info = info
        self.label = None
        self.set_label(label)
        self.prediction = prediction

    def has_action(self):
        return self.action is not None or 'frame/action/real_action' in self.info

    def get_label(self):
        return self.info.get('frame/action/label', self.label)

    def set_label(self, label):
        self.info['frame/action/label'] = label

    def was_blocked(self):
        return self.info.get('frame/action/should_block', False)

    def get_proposed_action(self):
        return self.info.get('frame/action/proposed_action', self.action)

    def get_real_action(self):
        return self.info.get('frame/action/real_action', self.action)

    def get_action(self, action_type):
        if action_type == Frame.REAL:
            return self.get_real_action()
        elif action_type == Frame.PROPOSED:
            return self.get_proposed_action()
        assert False

    def set_proposed_action(self, action):
        self.info['frame/action/proposed_action'] = action

    def set_real_action(self, action):
        self.info['frame/action/real_action'] = action
        self.action = action

    def __repr__(self):
        """ Doesn't print `self.image` or `self.observation` (too big)"""
        pretty_printed = """Action: {action},
        Reward: {reward}, Info: {info},
        Label: {label}, Prediction: {pred}""".format(
            action=self.action,
            reward=self.reward,
            info=self.info,
            label=self.label,
            pred=self.prediction)
        return pretty_printed

    def __setstate__(self, state):
        """ This method controls backwards-compatibility for unpickling: it's called every
        time a Frame is unpickled, so we can handle class-format changes easily."""
        if 'label' not in state.keys():
            self.__dict__.update(label=None)
        if 'prediction' not in state.keys():
            self.__dict__.update(prediction=None)
        self.__dict__.update(state)


class HumanLabelWrapper(gym.Wrapper):
    def __init__(self,
                 env_id,
                 env,
                 online_blocking_mode="action_replacement",
                 reward_scale=1.0,
                 catastrophe_reward=0.0,
                 **_):
        super().__init__(env)
        self.env_id = env_id
        self.conn = None
        self.current_frame = None
        self.episode_info_common = {}
        self.online_blocking_mode = online_blocking_mode
        self.catastrophe_reward = catastrophe_reward
        self.reward_scale = reward_scale
        self.episode_block_count = 0

        if isinstance(self.env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
            self.episode_info_common["action_set"] = self.env.unwrapped._action_set

        # todo(girish): assumes human_feedback.py is running and listening on port 6666

        logger.info('Opening connection...')
        # open connection on 6666 (won't support multiple workers)
        address = ('localhost', 6666)
        self.conn = Client(address, authkey=b'no-catastrophes-allowed')
        time.sleep(2.0)

    def _reset(self):
        self.__send_init_msg()
        observation = self.env.reset()
        # note - when run with FrameSaverWrapper, render happens twice (not sure if cached)
        image = self.env.render("rgb_array")
        self.current_frame = Frame(observation, None, 0, image, {})
        return observation

    def __send_init_msg(self):
        logger.info('Sending initial message...')
        init_msg_dict = {
            'msg': 'init'
        }  # todo - don't get episode num in human_feedback.py from this dict
        init_msg_dict.update(self.episode_info_common)
        init_msg_dict['env_id'] = self.env_id
        self.conn.send(init_msg_dict)

    def __send_frame_to_human(self, frame, proposed_action):
        """ Sends a frame with a proposed action to the human for feedback. """
        # if proposed_action is not None and frame.action is not None:
        frame.action = proposed_action
        frame.info["frame/action/proposed_action"] = proposed_action
        # info["frame/action/real_action"] = real_action

        proposed_action_meaning = ACTION_MEANING[self.episode_info_common["action_set"][
            proposed_action]]
        logger.info('Sending frame. Proposed action: {}'.format(proposed_action_meaning))
        self.conn.send(frame)

    def __handle_feedback(self, msg, proposed_action):
        """Given the proposed action and the human feedback message

        Returns:
            - a tuple (action, label).

        """
        logger.info('online step, received: {}'.format(msg))
        real_action = proposed_action
        label = None
        # parse feedback message
        if isinstance(msg, dict):
            label = 'b'
            if self.online_blocking_mode == "action_pruning":
                return None, label
            real_action = msg['action']
            proposed_action_meaning = ACTION_MEANING[self.episode_info_common["action_set"][
                proposed_action]]
            real_action_meaning = ACTION_MEANING[self.episode_info_common["action_set"][
                real_action]]
            logger.info('proposed action {} was blocked, replacing with action {}'
                        .format(proposed_action_meaning, real_action_meaning))
        return real_action, label

    def _step(self, proposed_action):
        self.__send_frame_to_human(self.current_frame, proposed_action=proposed_action)
        if not self.conn.poll(100.0):
            raise EnvironmentError("Failed to receive message!")
        msg = self.conn.recv()
        real_action, label = self.__handle_feedback(msg, proposed_action)
        if real_action is None:
            logger.info('Pruning action')
            observation, reward, done, info = self.current_frame.observation, \
                                              self.catastrophe_reward / self.reward_scale, \
                                              False, \
                                              self.current_frame.info
            info["frame/reward"] = reward
            info["frame/action/proposed_action"] = proposed_action
            info["frame/action/real_action"] = real_action
        else:
            logger.info(
                'Taking action: {}'
                .format(ACTION_MEANING[self.episode_info_common["action_set"][real_action]]))

            observation, reward, done, info = self.env.step(real_action)
            image = self.env.render("rgb_array")

            if real_action is not proposed_action:
                reward += self.catastrophe_reward / self.reward_scale
                logger.info('Action was changed.')
                self.episode_block_count += 1
                logger.info('Updated episode block count to {}'.format(self.episode_block_count))
                # self.__send_frame_to_human(self.current_frame, proposed_action=real_action)
                # self.conn.recv()
            info["frame/reward"] = reward
            # initialize current frame for next step
            self.current_frame = Frame(observation, None, reward, image, info.copy())
            info["frame/action/proposed_action"] = proposed_action
            info["frame/action/real_action"] = real_action

        if done:
            logger.info('Sending done message')
            self.conn.send({'msg': 'done'})
            info["global/episode_block_count"] = self.episode_block_count
            self.episode_block_count = 0

        info["frame/action/label"] = label
        return observation, reward, done, info

    def _close(self):
        if self.conn is not None:
            print('Closing connection on {host}:{port}'.format(host='localhost', port=6666))
            self.conn.send({'msg': 'close'})
            self.conn.close()


# TODO - save episodes consistently (last saved ep num + 1)
class FrameSaverWrapper(gym.Wrapper):
    def __init__(self, env, directory, max_episodes=1000):
        super().__init__(env)
        self.episode = Episode()
        self.log_coordinator = log_coordinator.LogCoordinator(
            max_length=max_episodes, logdir=directory, filename_str="e{i}.pkl.gz")
        self.next_frame = None
        self.episode_info_common = {}
        if isinstance(self.env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
            self.episode_info_common["action_set"] = self.env.unwrapped._action_set

    def _reset(self):
        observation = self.env.reset()
        if self.log_coordinator.should_log():
            image = self.env.render("rgb_array")
            self.next_frame = Frame(observation, None, 0, image, {})
        else:
            self.next_frame = None
        return observation

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.log_coordinator.should_log():
            current_frame = self.next_frame
            current_frame.action = action
            for key in info:
                if key.startswith("frame/action/"):
                    current_frame.info[key] = info[key]
            self.episode.frames.append(current_frame)

            info = {k: v for (k, v) in info.items() if not k.startswith("frame/action/")}
            image = self.env.render("rgb_array")
            self.next_frame = Frame(observation, None, reward, image, info)
        if done:
            if self.log_coordinator.should_log():
                self.episode.frames.append(self.next_frame)
                for key in info:
                    if key.startswith("frame/"):
                        self.episode.info[key] = info[key]
                self.episode.info.update(self.episode_info_common)
                self.episode.info["episode_num"] = self.log_coordinator.i
                filename = self.log_coordinator.get_filename()
                print('Saving to {}'.format(filename))
                save_episode(filename, self.episode)
                self.episode = Episode()
            self.log_coordinator.step()
        info = {key: value for key, value in info.items() if not key.startswith("frame/")}
        return observation, reward, done, info


def get_episode_number(filename):
    s = os.path.splitext(os.path.basename(filename))[0]
    s = os.path.splitext(s)[0]
    if s[0] == "e":
        s = s[1:]
    elif s[0] == "w":
        i = s.find("e")
        if i == -1:
            return -1
        else:
            s = s[i + 1:]
    return int(s)


def paths_with_suffix(directory, empty_ok=False, suffixes=("pkl.gz", )):
    if any(directory.endswith(s) for s in suffixes):
        return [directory]
    episode_paths = []
    for root, _, files in os.walk(directory):
        for name in files:
            if any(name.endswith(s) for s in suffixes):
                episode_paths.append(os.path.join(root, name))
    episode_paths = sorted(episode_paths, key=get_episode_number)
    assert empty_ok or episode_paths, "episode_paths empty:{}".format(directory)
    return episode_paths


def episode_paths(directory, empty_ok=False):
    return paths_with_suffix(directory, empty_ok, suffixes=("pkl.gz", ))


def feature_file_paths(directory, empty_ok=False):
    return paths_with_suffix(directory, empty_ok, suffixes=("features", ))


def load_episode(episode_path):
    try:
        with gzip.open(episode_path, "rb") as f:
            episode = pickle.load(f)
            episode.path = episode_path
            return episode
    except EOFError:
        print("Error reading: {}".format(episode_path))
        os.remove(episode_path)
        return None


def save_episode(episode_path, episode):
    with gzip.open(episode_path, "wb") as f:
        pickler = pickle.Pickler(f)
        pickler.dump(episode)


def check_episode(episode_path):
    try:
        episode = load_episode(episode_path)
    except EOFError:
        print("Error reading: {}".format(episode_path))
        os.remove(episode_path)
        return
    if not isinstance(episode.frames[0].action, (int, np.int64)):
        print("Episode Corrupted: {}".format(episode_path))
        os.remove(episode_path)
        print(type(episode.frames[0].action))


def fix_episode(episode_path):
    try:
        episode = load_episode(episode_path)
    except EOFError:
        print("Error reading: {}".format(episode_path))
        os.remove(episode_path)
        return
    if episode.version == 2:
        print("Version 2 already: {}".format(episode_path))
        return

    old_frames = episode.frames
    episode.frames = []
    for i in range(len(old_frames) - 1):
        f = old_frames[i]
        f.action = old_frames[i + 1].action
        episode.frames.append(f)

    episode.version = 2

    s = pickle.dumps(episode)
    with gzip.open(episode_path, "wb") as f:
        f.write(s)

        # pickler = pickle.Pickler(f)
        # pickler.dump(episode)

    # save_episode(episode_path, episode)


if __name__ == "__main__":
    args = parser.parse_args()

    env = gym.make("PongDeterministic-v3")
    env = FrameSaverWrapper(env, args.frames_dir, record_interval=1, max_episodes=100)
    for _ in range(100):
        observation = env.reset()
        for _ in range(10000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
    episode_paths = episode_paths(args.frames_dir)
    episodes = [load_episode(episode_path) for episode_path in episode_paths]
    print("Number of episodes: {}".format(len(episodes)))
