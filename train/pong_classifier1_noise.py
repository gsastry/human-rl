import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"humanrl"))

if __name__  == "__main__":
    from humanrl.classifier_tf import * # isort:skip

    import argparse
    import multiprocessing

    import numpy as np

    from humanrl import pong_catastrophe

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")

    episode_paths = frame.episode_paths("logs/PongDeterministic-v3-Episodes")
    np.random.seed(seed=42)
    np.random.shuffle(episode_paths)
    data_loader = DataLoader(pong_catastrophe.PongClassifierLabeller())
    datasets = data_loader.split_episodes(episode_paths, 1, 1, 1, seed=42)
    common_hparams = dict(use_action=False)

    args = parser.parse_args()
    if args.logdir == "":
        logdir = get_unused_logdir("models/tmp/pong1classifier")
        print(logdir)
    else:
        logdir = args.logdir
    hparams_list = [
        dict(batch_size=1024),
        # dict(image_crop_region=((34,34+160),(0,160)), batch_size=1),
        # dict(convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=1),
        # dict(image_crop_region=((34,34+160),(0,160)), convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=1),
        # dict(batch_size=5),
        # dict(image_crop_region=((34,34+160),(0,160)), batch_size=5),
        # dict(convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=5),
        # dict(image_crop_region=((34,34+160),(0,160)), convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5, batch_size=5),
        #
    ]

    run_experiments(logdir, data_loader, datasets, common_hparams, hparams_list,
                    steps=500, log_every=10)
                    # X_extra=features, y_extra=labels)
