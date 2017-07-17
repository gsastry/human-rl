import logging
import os
import time
from collections import defaultdict

import gym
import gym.envs.atari
import numpy as np
from gym import spaces
from gym.spaces.box import Box

import cv2
import universe
#import gym_ple
from humanrl import pacman, pong_catastrophe
from humanrl.catastrophe_wrapper import CatastropheWrapper
from humanrl.classifier_tf import (SavedCatastropheBlockerTensorflow,
                                   SavedCatastropheClassifierTensorflow)
from humanrl.exploration import ExplorationWrapper
from humanrl.location_wrapper import LocationWrapper
from humanrl.space_invaders import (SpaceInvadersMonitoringWrapper,
                                    SpaceInvadersStartsWrapper,
                                    SpaceInvadersStartsWrapper2)
from humanrl.utils import ACTION_MEANING, ACTION_MEANING_TO_ACTION
from universe import spaces as vnc_spaces
from universe import vectorized
from universe.spaces.vnc_event import keycode
from universe.wrappers import (BlockingReset, EpisodeID, GymCoreAction, Logger,
                               Unvectorize, Vectorize, Vision)

os.environ['SDL_VIDEODRIVER'] = "dummy"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()


def add_diagnostics(env, *args, **kwargs):
    env = Vectorize(env)
    env = DiagnosticsInfo(env, *args, **kwargs)
    env = Unvectorize(env)
    return env


class FireProofWrapper(gym.Wrapper):
    def _step(self, action):
        if "FIRE" in ACTION_MEANING[action]:
            action = ACTION_MEANING_TO_ACTION["NOOP"]
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


def create_env(env_id,
               client_id=None,
               remotes=None,
               explore=False,
               classifier_file="",
               blocker_file="",
               classifier_threshold=None,
               blocker_threshold=None,
               catastrophe_type="",
               log_reward=False,
               no_jump=False,
               extra_wrapper="",
               **kwargs):
    print('classifier_file, blocker_file, catastrophe_type: ', classifier_file, blocker_file,
          catastrophe_type)
    logger.info(str((classifier_file, blocker_file, catastrophe_type)))
    if env_id == "Pong":
        frame = ((34, 160 + 34), (0, 160), )
    if env_id == "Montezuma":
        env_id += "Revenge"
        frame = ((34, 160 + 34), (0, 160), )
    elif env_id == "Pacman":
        env = pacman.Pacman(**kwargs)
        return add_diagnostics(env)
    elif env_id == "RoadRunner":
        frame = ((40, 200), (0, 160), )
    elif env_id == "Berzerk":
        frame = ((0, 182), (0, 160), )
    elif env_id == "SpaceInvaders":
        frame = ((0, 200), (0, 160), )
    else:
        print("Unknown env id " + env_id)

    env = make_env(env_id, **kwargs)

    if env_id == "RoadRunner":
        if no_jump:
            env._action_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            env.action_space = spaces.Discrete(len(env._action_set))
            env = FireProofWrapper(env)
        env = RoadRunnerLevelWrapper(env, **kwargs)

    if env_id == "SpaceInvaders":
        if extra_wrapper == "SpaceInvadersStartsWrapper":
            env = SpaceInvadersStartsWrapper(env)
        if extra_wrapper == "SpaceInvadersStartsWrapper2":
            env = SpaceInvadersStartsWrapper2(env)
        env = SpaceInvadersMonitoringWrapper(env)

    if catastrophe_type:
        classifier = None
        if classifier_file and not (isinstance(classifier_file, list) and
                                    len(classifier_file) == 1 and not classifier_file[0]):
            classifier = SavedCatastropheClassifierTensorflow(
                classifier_file, threshold=classifier_threshold)

        blocker = None
        if blocker_file and not (isinstance(blocker_file, list) and len(blocker_file) == 1 and
                                 not blocker_file[0]):
            blocker = SavedCatastropheBlockerTensorflow(
                blocker_file, env.action_space.n, threshold=blocker_threshold)
        classifier_baseline = None

        allowed_actions_heuristic = None
        safe_action_mapping = None
        if env_id == "Pong":
            if catastrophe_type == "1":
                classifier_baseline = pong_catastrophe.CatastropheClassifierHeuristic(**kwargs)
                location = "bottom"
                if "location" in kwargs:
                    location = kwargs["location"]
                allowed_actions_heuristic = lambda observation: pong_catastrophe.allowed_actions_heuristic(observation, location=location)
                if classifier is None:
                    classifier = classifier_baseline
                if blocker is None:
                    blocker = pong_catastrophe.CatastropheBlockerHeuristic(**kwargs)

        if env_id == "RoadRunner":
            if catastrophe_type == "2":
                allowed_actions_heuristic = lambda observation: [4]  # Left

        if env_id == "SpaceInvaders":
            if catastrophe_type == "1":
                safe_action_mapping = "SpaceInvaders"

        env = CatastropheWrapper(
            env,
            classifier=classifier,
            blocker=blocker,
            classifier_baseline=classifier_baseline,
            allowed_actions_heuristic=allowed_actions_heuristic,
            safe_action_mapping=safe_action_mapping,
            **kwargs)

    if log_reward:
        env = LogReward(env)
    if explore:
        env = LocationWrapper(env)
        env = ExplorationWrapper(env, **kwargs)
    env = AtariEnvironment(env, frame_coordinates=frame, **kwargs)
    env = add_diagnostics(env, info_ignore_prefix=["ale."])
    env = LogActionFrequency(env)

    return env

    # spec = gym.spec(env_id)
    # atari = spec.tags.get('atari', False)
    #
    # if spec.tags.get('flashgames', False):
    #     return create_flash_env(env_id, client_id, remotes, **kwargs)
    # elif atari and spec.tags.get('vnc', False):
    #     return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    # elif spec.tags.get('ple'):
    #     return create_generic_env(env_id)
    # else:
    #     # Assume atari.
    #     assert "." not in env_id  # universe environments have dots in names.
    #
    #     return create_generic_env(env_id, atari=True)


def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    if env_id == 'flashgames.NeonRace-v0':
        # Better key space for this game.
        keys = ['left', 'right', 'up', 'left up', 'right up', 'down', 'up x']
    logger.info('create_flash_env(%s): keys=%s', env_id, keys)

    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    env.configure(
        fps=5.0,
        remotes=remotes,
        start_timeout=15 * 60,
        client_id=client_id,
        vnc_driver='go',
        vnc_kwargs={
            'encoding': 'tight',
            'compress_level': 0,
            'fine_quality_level': 50,
            'subsample_level': 3
        })
    return env


def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env


def create_atari_env(env_id):
    return create_generic_env(env_id, atari=True)


def wrap_generic_env(env, atari=False):
    env = Vectorize(env)
    if atari:
        env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env


def create_generic_env(env_id, atari=False):
    env = gym.make(env_id)
    env = Vectorize(env)
    if atari:
        env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env


def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)


class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503, info_ignore_prefix=[]):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._real_episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1
        self._info_ignore_prefix = info_ignore_prefix

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._real_episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info[
                    "stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info[
                    "stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        for k, v in info.items():
            if not any([k.startswith(prefix) for prefix in self._info_ignore_prefix]):
                to_log[k] = v
            # if k.startswith("log/") or k.startswith("frame/"):
            #     to_log[k] = v

        try:
            real_reward = info.pop("log/reward")
            self._real_episode_reward += real_reward
        except KeyError:
            pass

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s',
                        self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/real_episode_reward"] = self._real_episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/reward_per_step"] = self._episode_reward / self._episode_length
            to_log["global/episode_time"] = total_time

            for key, value in info.items():
                if not any([key.startswith(prefix) for prefix in self._info_ignore_prefix]):
                    to_log[key] = value

                # if key.startswith("global/") or key.startswith("action_frequency/"):
                #     to_log[key] = value

            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame


class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n


class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys

    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]

    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """

    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(
                    vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]


class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """

    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [
            ob[self.top:self.top + self.height, self.left:self.left + self.width, :]
            if ob is not None else None for ob in observation_n
        ]


def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame


class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [128, 200, 1])

    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]


class RewardScale(gym.Wrapper):
    def __init__(self, env, scale):
        super(RewardScale, self).__init__(env)
        self.scale = scale

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward *= self.scale
        return obs, reward, done, info


class LogReward(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['log/reward'] = reward
        return obs, reward, done, info


class LogActionFrequency(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_set = None
        if isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv):
            self._action_set = env.unwrapped._action_set
        self.counts = defaultdict(lambda: 0)

    def _reset(self):
        obs = self.env.reset()
        self.counts = defaultdict(lambda: 0)
        return obs

    def _action_name(self, action):
        if self._action_set is not None:
            action = self._action_set[action]
        return gym.envs.atari.atari_env.ACTION_MEANING[action]

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.counts[action] += 1
        if done:
            s = sum(self.counts.values())
            for a, c in self.counts.items():
                info["action_frequency/{}".format(self._action_name(a))] = float(c) / float(s)
        return obs, reward, done, info


def make_env(env_id, deterministic=True, v=3, **unused):
    if deterministic:
        env_id += "Deterministic"
    env_id += "-v%d" % v

    return gym.make(env_id)


class AtariEnvironment(gym.Wrapper):
    def __init__(self,
                 env,
                 frame_coordinates,
                 reward_scale=1.0,
                 death_penalty=0.0,
                 squash_rewards=False,
                 **_):
        super(AtariEnvironment, self).__init__(env)
        self.reward_scale = reward_scale
        self.frame_coordinates = frame_coordinates
        self.observation_space = gym.spaces.Box(0.0, 1.0, [42, 42, 1])
        self.death_penalty = death_penalty
        self.lives = None
        self.squash_rewards = squash_rewards

    def process_observation(self, obs):
        #         return obs
        frame = obs.mean(2)
        (x1, x2), (y1, y2) = self.frame_coordinates
        frame = frame[x1:x2, y1:y2]
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [42, 42, 1])
        return frame

    def _reset(self):
        obs = self.env.reset()
        self.lives = None
        return self.process_observation(obs)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.process_observation(obs)

        if self.squash_rewards:
            reward = float(np.sign(reward))
        else:
            reward = float(reward) / float(self.reward_scale)

        info["frame/lives"] = info["ale.lives"]
        if self.lives is None:
            self.lives = info["ale.lives"]
        else:
            current_lives = info["ale.lives"]
            lost = self.lives - current_lives
            self.lives = current_lives
            if lost > 0:
                reward -= lost * self.death_penalty

        return obs, reward, done, info


class RoadRunnerLevelWrapper(gym.Wrapper):
    def __init__(self, env, lose_first_two_lives=False, max_level=None, **_):
        super().__init__(env)
        self.level = None
        self.in_level_transition = False
        self.higher_level_reward = 0
        self.higher_level_steps = 0
        self.level_1_lives_lost = 0
        self.lives = None
        self.lose_first_two_lives = lose_first_two_lives
        self.max_level = max_level

    def _reset(self):
        obs = self.env.reset()
        self.level = 1
        self.in_level_transition = False
        self.higher_level_reward = 0
        self.higher_level_steps = 0
        self.level_1_lives_lost = 0
        self.lives = None
        if self.lose_first_two_lives:
            while (self.lives != 1):
                self.step(0)
        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.lives is None:
            self.lives = info["ale.lives"]
        else:
            current_lives = info["ale.lives"]
            lost = self.lives - current_lives
            self.lives = current_lives
            if lost > 0 and self.level == 1:
                self.level_1_lives_lost += 1

        if np.mean(obs) < 10:
            if not self.in_level_transition:
                self.in_level_transition = True
                self.level += 1
        else:
            self.in_level_transition = False

        if self.level > 1:
            self.higher_level_reward += reward
            self.higher_level_steps += 1

        if self.max_level is not None and self.level > self.max_level:
            done = True

        info["frame/level"] = self.level
        if done:
            info["global/higher_level_reward"] = self.higher_level_reward
            info["global/higher_level_steps"] = self.higher_level_steps
            info["global/level_1_lives_lost"] = self.level_1_lives_lost

        return obs, reward, done, info
