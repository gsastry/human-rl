import argparse
import datetime
import gzip
import itertools
import logging
import os
import os.path
import pickle
import random
import time
from multiprocessing.connection import Listener
from subprocess import call

import numpy as np
from PIL import Image

import cv2
import humanrl.frame as frame_module
from humanrl.utils import (ACTION_MEANING, ACTION_MEANING_TO_ACTION,
                           SAFE_ACTION_MAPPINGS, identity, shuffle)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument(
    '-f',
    '--frames-dir',
    type=str,
    default="/tmp/pong/episodes",
    help="Directory to read and write labeled frames to")
parser.add_argument('-o', '--output-dir', type=str, default="", help="Directory to write frames to")
parser.add_argument(
    '-n',
    '--dry-run',
    action='store_true',
    help="Will show the video, but won't store human labels")
parser.add_argument(
    '--reversed',
    action='store_true',
    help="go through episodes in reverse order (to see more recent ones first)")
parser.add_argument('--random', action='store_true', help="go through episodes in random order")
parser.add_argument(
    '--image_scale', '-i', type=float, default=2.0, help="factor to scale up image by")
parser.add_argument('--env-id', '-e', type=str, default='', help="Environment Id")
parser.add_argument('--pause', '-p', action='store_true', help="pause at start of episode")
parser.add_argument(
    '--label_mode',
    '-m',
    choices=["catastrophe", "block"],
    default="catastrophe"
    "Select which type of labels will be set during this session, which also "
    "controls what label the next/previous keys look for.")
parser.add_argument('--online', action='store_true', help="online mode")
parser.add_argument('--safe_action', type=int, default=2)
parser.add_argument('--safe_action_mapping', type=str, default="")
parser.add_argument(
    '--blocking_mode',
    choices=["action_replacement", "action_pruning"],
    default="action_replacement")

IMAGE_SCALE_UP = 2.


def rgb_to_bgr(img):
    """
    Args:
        img (numpy array of shape (width, height, 3): input image of rgb values to convert to bgr values

    Summary:
        Convert an rgb image to a bgr image. OpenCV2 requires BGR, but Gym/vnc save in RGB.
    """
    rgb = img[..., ::-1].copy()  # no idea why without copy(), get a cv2 error later
    return rgb


def save_labels(filename, episode, frames):
    """
    Args:
        filename (str): file to save labeled episode to.
        episode (episode): an episode to save labeled `frames` to (this is pickled and gzipped)
        frames (list of frames): a time ordered list of labeled frames.

    Summary:
        Save labeled frames to a file.
    """

    episode.frames = frames
    with gzip.open(filename, "wb") as f:
        pickler = pickle.Pickler(f)
        pickler.dump(episode)


def add_text(img, text, text_top, image_scale):
    """
    Args:
        img (numpy array of shape (width, height, 3): input image
        text (str): text to add to image
        text_top (int): location of top text to add
        image_scale (float): image resize scale

    Summary:
        Add display text to a frame.

    Returns:
        Next available location of top text (allows for chaining this function)
    """
    cv2.putText(
        img=img,
        text=text,
        org=(0, text_top),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.15 * image_scale,
        color=(255, 255, 255))
    return text_top + int(5 * image_scale)


def decorate_img_for_env(img, env_id, image_scale):
    """
    Args:
        img (numpy array of (width, height, 3)): input image
        env_id (str): the gym env id
        image_scale (float): a scale to resize the image

    Returns:
        an image

    Summary:
        Adds environment specific image decorations. Currently used to make it easier to
        block/label in Pong.

    """
    if env_id is not None and 'Pong' in env_id:
        h, w, _ = img.shape
        est_catastrophe_y = h - 142
        est_block_clearance_y = est_catastrophe_y - int(20 * image_scale)
        # cv2.line(img, (0, est_catastrophe_y), (int(500 * image_scale), est_catastrophe_y), (0, 0, 255))
        cv2.line(img, (250, est_catastrophe_y), (int(500 * image_scale), est_catastrophe_y), (0, 255, 255))
        # cv2.line(img, (0, est_block_clearance_y), (int(500 * image_scale), est_block_clearance_y),
        #          (255, 0, 0))
    return img


def render(frame, action_set, env_id='Pong', extra_text=None, image_scale=2.0, prev_actions=None):
    """
    Args:
        frame (a `Frame`): input frame
        action_set (list of ints): action set for the environment
        env_id (str): Gym environment id
        image_scale (float): image resize scale
        prev_actions (list of ints): list of previous actions to display

    Summary:
        Render a frame with various helper texts and adornments.

    Returns:
        a rendered image (numpy array of shape (width, height, 3))
    """
    if extra_text is None:
        extra_text = []
    if prev_actions is None:
        prev_actions = []
    img = rgb_to_bgr(frame.image)
    img = np.pad(img, ((0, 40), (0, 0), (0, 0)), 'constant', constant_values=0)
    label = frame.get_label()
    img = cv2.resize(img, (0, 0), fx=image_scale, fy=image_scale)
    if label == 'c':
        cv2.line(img, (0, 0), (int(500 * image_scale), 0), (0, 0, 255), 20)
    elif label == 'b' or frame.was_blocked():
        cv2.line(img, (0, 0), (int(500 * image_scale), 0), (255, 0, 0), 20)

    text_top = int(.86 * img.shape[0])
    for text in extra_text:
        text_top = add_text(img, text, text_top, image_scale=image_scale)
    # text_top = add_text(img, "Mean: {}".format(np.mean(frame.image)), text_top, image_scale=image_scale)
    if "frame/action/proposed_action" in frame.info:
        # print("Proposed Action: {}".format(ACTION_MEANING[action_set[frame.action]]))
        text_top = add_text(
            img,
            "Proposed Action: {}".format(
                ACTION_MEANING[action_set[frame.get_proposed_action()]]),
            text_top,
            image_scale=image_scale)
    if "frame/action/real_action" in frame.info and frame.get_real_action() is not None:
        # print("Real Action: {}".format(ACTION_MEANING[action_set[frame.get_real_action()]]))
        text_top = add_text(
            img,
            "Real Action: {}".format(
                ACTION_MEANING[action_set[frame.get_real_action()]]),
            text_top,
            image_scale=image_scale)
    if frame.prediction is not None:
        text_top = add_text(
            img, "Prediction: {:.2e}".format(frame.prediction), text_top, image_scale=image_scale)
    # Display the previous two actions
    for idx, a in enumerate(reversed(prev_actions)):
        if a is not None:
            text_top = add_text(
                img,
                "Previous Action {}: {}".format(idx, ACTION_MEANING[action_set[a]]),
                text_top,
                image_scale=image_scale)
    for k in sorted(frame.info.keys()):
        if k.startswith("frame/"):
            text_top = add_text(
                img, "{}: {}".format(k, frame.info[k]), text_top, image_scale=image_scale)
    return decorate_img_for_env(img, env_id, image_scale)


def offline_episode_feedback_setup(label_mode, frames_in_ep):
    """Set up necessary variables for offline labeling/viewing.

    Arguments:
        label_mode: 'catastrophe', 'block', or any string
        frames_in_ep: List[Frame]
    """
    frames_in_ep = frames_in_ep[:]

    if label_mode == "catastrophe":
        labels = np.array([frame.get_label() == 'c' for frame in frames_in_ep])
    elif label_mode == "block":
        labels = np.array([frame.get_label() == 'b' for frame in frames_in_ep])
    else:
        labels = np.array([False] * len(frames_in_ep))

    predictions = np.array([(frame.prediction if frame.prediction is not None else -1.0)
                            for frame in frames_in_ep])

    false_negative_loss = labels * (1.0 - predictions)
    false_positive_loss = (1 - labels) * predictions
    false_negative_ind = np.argsort(-false_negative_loss)
    false_positive_ind = np.argsort(-false_positive_loss)

    false_negative_pos = -1
    false_positive_pos = -1

    death_ind = []
    for i in range(len(frames_in_ep) - 1):
        current_lives = frames_in_ep[i].info.get("frame/lives")
        next_lives = frames_in_ep[i + 1].info.get("frame/lives")
        if current_lives is not None and next_lives is not None and current_lives > next_lives:
            death_ind.append(i)
    death_pos = -1

    return false_negative_loss, \
           false_positive_loss, \
           false_negative_ind, \
           false_positive_ind, \
           false_negative_pos, \
           false_positive_pos, \
           death_ind, \
           death_pos


def setup_for_ep(episode_path, episode, output_dir=None, dry_run=True):
    """Handles setup for each episode.
    Returns:
        tuple (output_filename, action_set, should_skip)
        """

    should_skip = False
    if output_dir:
        components = os.path.normpath(episode_path).split(os.path.sep)
        f = components[-1]
        print(components)
        if len(components) > 1 and components[-2][0] == 'w' and components[-2][1:].isnumeric():
            f = "{}_{}".format(components[-2], components[-1])
        output_filename = os.path.join(output_dir, f)
        if dry_run and os.path.exists(output_filename):
            should_skip = True
    else:
        output_filename = episode_path

    action_set = list(range(18))
    if "Pong" in episode_path or "SpaceInvaders" in episode_path:
        action_set = [0, 1, 3, 4, 11, 12]
    if "action_set" in episode.info:
        action_set = episode.info["action_set"]
    return output_filename, action_set, should_skip


class Episodes(object):
    """
    An iterator to read episodes from a directory.

    This assumes episodes are stored in the same format as `Frame.py` writes them.

    """

    def __init__(self, episodes_dir, transform_func=identity):
        self.episode_paths = []
        if episodes_dir:
            self.episode_paths = transform_func(frame_module.episode_paths(episodes_dir))

    def __iter__(self):
        for episode_path in self.episode_paths:
            episode = frame_module.load_episode(episode_path)
            yield episode_path, episode, 0  # reset the frame_index


class OnlineEpisodes(Episodes):
    """An empty infinite iterator. """

    def __init__(self):
        super(OnlineEpisodes, self).__init__(episodes_dir=None)

    def __iter__(self):
        return itertools.repeat([None, frame_module.Episode(), 0])


class ViewerState(object):
    """Handles state needed to display and navigate through frames.
    """

    def __init__(self, conn=None, output_dir=None, delay=500):
        self.current_frame = None
        self.prev_actions = [None, None]
        self.frame_index = 0

        self.env_id = None
        self.action_set = list(range(18))

        self.paused = False
        self.delay = delay
        self.EXIT = False
        self.save = False
        self.episode_num_in_session = 0
        self.skip_frame = False
        self.was_advanced = True

        # online only
        self.feedback_msg = 1
        self.conn = conn
        self.close = False
        self.save = False

        # offline only
        self.skip_episode = False

        self.output_dir = output_dir
        self.output_filename = None

    def advance(self, online=False):
        if online:
            self.conn.send(self.feedback_msg)
        self.prev_actions[0] = self.prev_actions[1]
        if self.current_frame is not None:
            self.prev_actions[1] = self.current_frame.get_real_action()
        self.frame_index += 1
        self.was_advanced = True

    def reset_for_frame(self):
        self.skip_frame = False
        self.close = False
        self.feedback_msg = 1
        self.was_advanced = False

    def reset_for_episode(self):
        self.skip_episode = False
        self.save = False
        self.current_frame = None


def online_receive_frame(viewer_state, episode):
    logger.info('Waiting for message...')
    if not viewer_state.conn.poll(1000.0):
        raise EnvironmentError("Failed to receive message!")
    msg = viewer_state.conn.recv()
    if isinstance(msg, frame_module.Frame):
        viewer_state.current_frame = msg
        episode.frames.append(viewer_state.current_frame)
    elif msg['msg'] == 'init':
        viewer_state.prev_frame = msg
        viewer_state.action_set = msg['action_set']
        viewer_state.episode_num_in_session += 1
        print('Initial message received')
        print('... got action_set: {}'.format(viewer_state.action_set))
        print('... episode_num: {}'.format(viewer_state.episode_num_in_session))
        viewer_state.env_id = msg['env_id']
        viewer_state.skip_frame = True
    elif msg['msg'] == 'close':
        viewer_state.conn.close()
        viewer_state.close = True
    elif msg['msg'] == 'done':
        viewer_state.frame_index = 0
        proceed = input('Proceed to next episode?')

        if proceed != 'y':
            viewer_state.EXIT = True

        while True:
            k = cv2.waitKey(viewer_state.delay) & 0xFF
            if k == 255: break
        viewer_state.skip_episode = True
    else:
        print('Unknown message received: {}'.format(msg))
        viewer_state.skip_frame = True

    return viewer_state, episode


def get_safe_action(action, args):
    if args.blocking_mode == "action_pruning":
        return None
    if args.safe_action_mapping:
        if args.safe_action_mapping not in SAFE_ACTION_MAPPINGS:
            raise ValueError("{} {}".format(args.safe_action_mapping, SAFE_ACTION_MAPPINGS.keys()))
        if action not in SAFE_ACTION_MAPPINGS[args.safe_action_mapping]:
            raise ValueError(
                "{} {}".format(action, SAFE_ACTION_MAPPINGS[args.safe_action_mapping].keys()))
        return SAFE_ACTION_MAPPINGS[args.safe_action_mapping][action]
    return args.safe_action


def main():
    random.seed()
    args = parser.parse_args()

    print("Displaying video...")

    K_FASTER = ord('1')
    K_SLOWER = ord('2')
    K_PAUSE = ord('3')
    K_BACK = ord('4')
    K_FWD = ord('5')

    K_PREV_FALSE_POSITIVE = ord('7')
    K_NEXT_FALSE_POSITIVE = ord('8')
    K_PREV_FALSE_NEGATIVE = ord('9')
    K_NEXT_FALSE_NEGATIVE = ord('0')

    K_PREV_LABEL = ord('u')
    K_NEXT_LABEL = ord('i')
    K_PREV_DEATH = ord('o')
    K_NEXT_DEATH = ord('p')

    K_CATASTROPHE = ord('c')
    K_BLOCK = ord('b')
    K_SAVE_VIDEO = ord('v')
    K_ESC = ord('\x1b')  # esc
    K_SKIP = ord('s')
    K_REMOVE_LABEL = ord('r')
    K_NONE = -1 & 0xFF

    online_key_set = frozenset([K_ESC, K_BLOCK, K_FASTER, K_SLOWER, K_NONE, K_PAUSE, K_FWD])

    # setup for offline mode
    if not args.online:
        episode_paths = frame_module.episode_paths(args.frames_dir)
        print("Episodes: {}".format(len(episode_paths)))
        if args.reversed:
            episode_paths = reversed(episode_paths)
        if args.random:
            random.shuffle(episode_paths)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    episodes_iter = None
    viewer_state = None
    if args.online:
        print('In online mode...')
        # listen to port for frames
        print('Waiting for connection...')
        address = ('localhost', 6666)
        listener = Listener(address, authkey=b'no-catastrophes-allowed')
        conn = listener.accept()
        print('Connection accepted from {}'.format(listener.last_accepted))

        viewer_state = ViewerState(conn=conn, output_dir=args.output_dir)
        episodes_iter = OnlineEpisodes()
    else:
        transform_func = identity
        if args.reversed:
            transform_func = reversed
        if args.random:
            transform_func = shuffle
        episodes_iter = Episodes(args.frames_dir, transform_func)
        viewer_state = ViewerState(output_dir=args.output_dir)

    for episode_path, episode, viewer_state.frame_index in episodes_iter:
        if episode is None:
            print('Error: no episode')
            continue
        if viewer_state.EXIT:
            break

        if not args.online:

            labels_check = np.array([frame.get_label() == 'b' for frame in episode.frames])
            if any(labels_check):
                print('FOUND SOME LABELS ------------------')

            # set up output_dir for episode
            viewer_state.output_filename, viewer_state.action_set, viewer_state.skip_episode = \
                setup_for_ep(episode_path, episode, output_dir=args.output_dir, dry_run=args.dry_run)

            false_negative_loss, \
            false_positive_loss, \
            false_negative_ind, \
            false_positive_ind, \
            false_negative_pos, \
            false_positive_pos, \
            death_ind, \
            death_pos = offline_episode_feedback_setup(args.label_mode, episode.frames)

            viewer_state.episode_num_in_session = episode.info.get('episode_num', 0)
            viewer_state.env_id = args.env_id if args.env_id else None

            print(viewer_state.episode_num_in_session)

        viewer_state.paused = args.pause
        viewer_state.reset_for_episode()
        frames = episode.frames
        last_status = None
        while not viewer_state.EXIT:
            viewer_state.frame_index = min(viewer_state.frame_index, len(frames) - 1)

            if args.online:
                if (viewer_state.current_frame is None or not viewer_state.paused or
                        viewer_state.was_advanced):
                    viewer_state.reset_for_frame()  # todo - check if works for offline
                    viewer_state, episode = online_receive_frame(
                        viewer_state=viewer_state, episode=episode)
            else:
                viewer_state.current_frame = frames[viewer_state.frame_index]

            if viewer_state.skip_frame:
                continue
            if viewer_state.skip_episode or viewer_state.close:
                break

            status = (viewer_state.frame_index, viewer_state.current_frame.get_proposed_action(),
                      viewer_state.was_advanced)
            if status != last_status:
                print(status)
                last_status = status

            img = render(
                viewer_state.current_frame,
                viewer_state.action_set,
                viewer_state.env_id,
                extra_text=[
                    "Episode: {}, Frame: {}".format(viewer_state.episode_num_in_session,
                                                    viewer_state.frame_index)
                ],
                image_scale=args.image_scale,
                prev_actions=viewer_state.prev_actions)

            cv2.imshow('frame', img)
            k = cv2.waitKey(viewer_state.delay) & 0xFF
            if args.online and k not in online_key_set:
                print("Key '{}' not supported in online mode; "
                      "if you want to support it, add it to the keyset.".format(chr(k)))
                if not viewer_state.paused:
                    viewer_state.advance(online=args.online)
            elif k == K_ESC:  # esc key
                viewer_state.EXIT = True
                if args.online:
                    print('Exiting...')
                    print('Killing A3C...')
                    call(['tmux', 'kill-session', '-t', 'a3c'])
                # if save:
                #     print("Writing episode {} to {}".format(episode_num, args.frames_dir))
                #     save_labels(directory=args.frames_dir, episode=episode, episode_num=episode_num, frames=frames)
                # cv2.destroyAllWindows()
            elif k == K_CATASTROPHE and args.label_mode == "catastrophe":  # 'c'
                print('catastrophe!')
                viewer_state.current_frame.set_label("c")
                viewer_state.save = True
                viewer_state.advance(online=args.online)
            elif k == K_BLOCK and args.label_mode == "block":  # 'b'
                print('blocking action')
                proposed_action = viewer_state.current_frame.get_proposed_action()
                safe_action = get_safe_action(viewer_state.current_frame.get_proposed_action(),
                                              args)
                if args.blocking_mode == "action_replacement" and proposed_action == safe_action:
                    print('not blocking, action is safe')
                else:
                    viewer_state.current_frame.set_label("b")
                    viewer_state.current_frame.set_real_action(safe_action)
                    if args.online:
                        viewer_state.feedback_msg = {
                            'feedback': 'b',
                            'action': viewer_state.current_frame.get_real_action()
                        }
                    viewer_state.save = True
                viewer_state.advance(online=args.online)
            elif k == K_FASTER:
                viewer_state.delay -= 50
                viewer_state.delay = max(1, viewer_state.delay)
                print('New delay: {}'.format(viewer_state.delay))
                if not viewer_state.paused:
                    viewer_state.advance(online=args.online)
            elif k == K_REMOVE_LABEL and not args.dry_run:
                print('removing catastrophe label...')
                viewer_state.current_frame.set_label(None)
                viewer_state.save = True
            elif k == K_SLOWER:
                viewer_state.delay += 50
                print('New delay: {}'.format(viewer_state.delay))
                if not viewer_state.paused:
                    viewer_state.advance(online=args.online)
            elif k == K_BACK:
                viewer_state.frame_index = max(0, viewer_state.frame_index - 1)
            elif k == K_FWD:
                viewer_state.advance(online=args.online)
            elif k == K_PREV_FALSE_POSITIVE and len(false_positive_ind) > 0:
                false_positive_pos = (false_positive_pos - 1) % len(false_positive_ind)
                viewer_state.frame_index = false_positive_ind[false_positive_pos]
            elif k == K_NEXT_FALSE_POSITIVE and len(false_positive_ind) > 0:
                false_positive_pos = (false_positive_pos + 1) % len(false_positive_ind)
                viewer_state.frame_index = false_positive_ind[false_positive_pos]
            elif k == K_PREV_FALSE_NEGATIVE and len(false_negative_ind) > 0:
                false_negative_pos = (false_negative_pos - 1) % len(false_negative_ind)
                viewer_state.frame_index = false_negative_ind[false_negative_pos]
            elif k == K_NEXT_FALSE_NEGATIVE and len(false_negative_ind) > 0:
                false_negative_pos = (false_negative_pos + 1) % len(false_negative_ind)
                viewer_state.frame_index = false_negative_ind[false_negative_pos]
            elif k == K_NEXT_LABEL:
                for j in range(len(frames))[viewer_state.frame_index + 1:]:
                    if frames[j].get_label() is not None:
                        print(j, frames)
                        viewer_state.frame_index = j
                        viewer_state.was_advanced = True
                        break
            elif k == K_PREV_LABEL:
                for j in reversed(range(len(frames))[:viewer_state.frame_index]):
                    if frames[j].get_label() is not None:
                        viewer_state.frame_index = j
                        viewer_state.was_advanced = True
                        break
            elif k == K_PREV_DEATH and len(death_ind) > 0:
                death_pos = (death_pos - 1) % len(death_ind)
                viewer_state.frame_index = death_ind[death_pos]
            elif k == K_NEXT_DEATH and len(death_ind) > 0:
                death_pos = (death_pos + 1) % len(death_ind)
                viewer_state.frame_index = death_ind[death_pos]
            elif k == K_SKIP:
                viewer_state.skip_episode = True
            elif k == K_PAUSE:
                viewer_state.paused = not viewer_state.paused
                if not viewer_state.paused:
                    viewer_state.advance(online=args.online)
            elif k == K_SAVE_VIDEO:
                video_output_filename = viewer_state.output_filename + ".avi"
                fourcc = cv2.VideoWriter_fourcc(* 'MJPG')
                img = render(frames[0], viewer_state.action_set, viewer_state.env_id)
                writer = cv2.VideoWriter(video_output_filename, fourcc, 20.0, (img.shape[1],
                                                                               img.shape[0]))
                print('Saving video file as {}'.format(video_output_filename))
                # out = cv2.VideoWriter(output_filename, fourcc, 20.0, img.shape)
                for i, f in enumerate(frames):
                    img = render(f, viewer_state.action_set, viewer_state.env_id, extra_text=[
                        "Episode: {}, Frame: {}".format(viewer_state.episode_num_in_session, i)
                    ])
                    writer.write(img)
                    cv2.imshow('frame', img)
                writer.release()
            else:
                if not viewer_state.paused:
                    viewer_state.advance(online=args.online)

        if viewer_state.save and not args.dry_run:
            if viewer_state.output_filename is not None:
                print('Saving to file: {}'.format(viewer_state.output_filename))
                save_labels(
                    filename=viewer_state.output_filename,
                    episode=episode,
                    # episode_num=viewer_state.episode_num_in_session,
                    frames=frames)
    viewer_state.EXIT = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
