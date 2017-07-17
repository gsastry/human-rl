import argparse
import os
import shutil
import sys

import numpy as np

from build_feature_files import build_feature_files
from humanrl import frame
from humanrl.classifier_tf import (DataLoader, HumanOnlineBlockerLabeller,
                                   TensorflowClassifierHparams,
                                   get_unused_logdir, run_experiments)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")
    parser.add_argument('--input_dir', type=str, default="labels/RoadRunnerOnline")
    parser.add_argument('--steps', type=int, default=20000)
    args = parser.parse_args()

    common_hparams = dict(
        use_action=True,
        batch_size=512,
        input_processes=8,
        image_shape=[160, 160, 3],
        image_crop_region=((40, 200), (0, 160)),
        expected_positive_weight=0.05)

    paths = frame.episode_paths(args.input_dir)

    data_loader = DataLoader(
        hparams=TensorflowClassifierHparams(**common_hparams),
        labeller=HumanOnlineBlockerLabeller())
    datasets = paths, paths, []
    # data_loader.split_episodes(paths, 1, 0, 0, use_all=True, seed=42)

    if args.logdir == "":
        logdir = get_unused_logdir("models/tmp/roadrunner/online/blocker")
    else:
        logdir = args.logdir
    hparams_list = [
        # dict(),
        # dict(label_smoothing=True),
        dict(
            convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6,
            fully_connected_stack_args=[20, 20],
            positive_weight_target=0.5,
            keep_prob=0.5),
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
