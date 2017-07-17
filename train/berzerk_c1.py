import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"humanrl"))

if __name__  == "__main__":
    import argparse

    import numpy as np

    from humanrl.classifier_tf import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")

    episode_paths = frame.episode_paths("labels/Berzerk/c1")
    data_loader = DataLoader(HumanLabeller(), hparams=TensorflowClassifierHparams())
    datasets = data_loader.split_episodes(episode_paths, 1, 1, 0, use_all=True)
    common_hparams = dict(use_action=False, batch_size=1)

    args = parser.parse_args()
    if args.logdir == "":
        logdir = get_unused_logdir("models/tmp/berzerk/c1/classifier")
    else:
        logdir = args.logdir
    hparams_list = [
        dict(convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 5,
             fully_connected_stack_args=[20, 20], keep_prob=0.5,
             ),
        dict(convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 5,
             fully_connected_stack_args=[20, 20], keep_prob=0.5,
             label_smoothing=True,
             ),
        dict(),
    ]

    run_experiments(logdir, data_loader, datasets, common_hparams, hparams_list,
                    steps=2000, log_every=100, predict_episodes=True)
