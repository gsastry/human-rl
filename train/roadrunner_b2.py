import argparse
import os
import shutil
import sys

import numpy as np

from build_feature_files import build_feature_files
from humanrl import frame
from humanrl.classifier_tf import (DataLoader, TensorflowClassifierHparams,
                                   get_unused_logdir, run_experiments)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))


class DeathsAndJumpsLabeller:
    def __init__(self, block_radius=1):
        self.block_radius = block_radius

    def label(self, features, episode):
        ACTION_MEANING = {
            None: "NONE",
            0: "NOOP",
            1: "FIRE",
            2: "UP",
            3: "RIGHT",
            4: "LEFT",
            5: "DOWN",
            6: "UPRIGHT",
            7: "UPLEFT",
            8: "DOWNRIGHT",
            9: "DOWNLEFT",
            10: "UPFIRE",
            11: "RIGHTFIRE",
            12: "LEFTFIRE",
            13: "DOWNFIRE",
            14: "UPRIGHTFIRE",
            15: "UPLEFTFIRE",
            16: "DOWNRIGHTFIRE",
            17: "DOWNLEFTFIRE",
        }
        labels = [False] * (len(episode.frames) - 1)
        for i in range(len(episode.frames) - 1):
            level = episode.frames[i].info.get("frame/level")
            if level is not None and level > 1:
                break
            current_lives = episode.frames[i].info.get("frame/lives")
            next_lives = episode.frames[i + 1].info.get("frame/lives")
            if (current_lives == 1 and next_lives == 0):
                for j in range(0, self.block_radius):
                    if j > i:
                        break
                    labels[i - j] = True
                for j in range(1, 10):
                    if j > i:
                        break
                    if "FIRE" in ACTION_MEANING[episode.frames[i - j].action]:
                        labels[i - j] = True

        labels = np.array(
            [label for frame, label in zip(episode.frames, labels) if frame.action is not None])
        return features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")
    parser.add_argument('--input_dir', type=str, default="logs/RoadRunner")
    parser.add_argument('--labels_dir', type=str, default="labels/RoadRunner/b2")
    parser.add_argument('--regenerate_feature_files', action="store_true")
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--block_radius', type=int, default=1)
    args = parser.parse_args()

    common_hparams = dict(
        use_action=True,
        batch_size=512,
        input_processes=8,
        image_shape=[160, 160, 3],
        expected_positive_weight=0.013,
        image_crop_region=((40, 200), (0, 160), ))

    num_episodes = 2000
    negative_example_keep_prob = 0.1
    if args.regenerate_feature_files and os.path.exists(args.labels_dir):
        shutil.rmtree(args.labels_dir, ignore_errors=True)

    if not os.path.exists(args.labels_dir):
        print("Writing feature files")
        data_loader = DataLoader(
            hparams=TensorflowClassifierHparams(**common_hparams),
            labeller=DeathsAndJumpsLabeller(args.block_radius))
        label_counts = build_feature_files(args.input_dir, args.labels_dir, data_loader,
                                           num_episodes, negative_example_keep_prob)
        print(label_counts)
    paths = frame.feature_file_paths(args.labels_dir)

    assert len(paths) > 0, "assert len(paths) > 0, {}, {}".format(len(paths), num_episodes)

    data_loader = DataLoader(hparams=TensorflowClassifierHparams(**common_hparams))
    datasets = data_loader.split_episodes(paths, 1, 1, 0, use_all=True, seed=42)

    if args.logdir == "":
        logdir = get_unused_logdir("models/tmp/roadrunner/c1/blocker")
    else:
        logdir = args.logdir
    hparams_list = [
        # dict(),
        # dict(label_smoothing=True),
        dict(
            convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6,
            fully_connected_stack_args=[20, 20],
            keep_prob=0.5,
            positive_weight_target=0.5),
    ]

    run_experiments(
        logdir,
        data_loader,
        datasets,
        common_hparams,
        hparams_list,
        steps=args.steps,
        log_every=100,
        predict_episodes=False)
