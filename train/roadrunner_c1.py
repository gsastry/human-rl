if __name__ == "__main__":
    import argparse

    import numpy as np

    from humanrl.classifier_tf import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")

    episode_paths = frame.episode_paths("labels/RoadRunnerRandom/c1")
    data_loader = DataLoader(
        HumanLabeller(), hparams=TensorflowClassifierHparams(
            image_shape=[210, 160, 3], ))
    datasets = data_loader.split_episodes(episode_paths, 1, 1, 0, use_all=True)
    common_hparams = dict(
        use_action=False, batch_size=512, input_processes=2, expected_positive_weight=0.35)

    args = parser.parse_args()
    if args.logdir == "":
        logdir = get_unused_logdir("models/tmp/roadrunner/c1/classifier")
    else:
        logdir = args.logdir
    hparams_list = [
        dict(
            convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6,
            fully_connected_stack_args=[20, 20],
            keep_prob=0.5,
            image_shape=[210, 160, 3],
            positive_weight_target=0.5),
    ]

    run_experiments(
        logdir,
        data_loader,
        datasets,
        common_hparams,
        hparams_list,
        steps=10,
        log_every=10,
        predict_episodes=False)
