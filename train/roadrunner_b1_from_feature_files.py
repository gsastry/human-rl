import os
import sys

from build_feature_files import build_feature_files

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

if __name__ == "__main__":
    import argparse

    from humanrl.classifier_tf import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")

    common_hparams = dict(
        use_action=True,
        batch_size=512,
        input_processes=2,
        image_shape=[210, 160, 3],
        expected_positive_weight=0.015, )

    base_directory = "logs/RoadRunner"
    new_directory = "labels/RoadRunner/b1/3"
    num_episodes = 100
    negative_example_keep_prob = 0.1

    if not os.path.exists(new_directory):
        print("Writing feature files")
        classifier = SavedCatastropheClassifierTensorflow(
            "models/roadrunner/c1/classifier/final.ckpt")
        data_loader = DataLoader(
            hparams=classifier.classifier.hparams, labeller=BlockerLabeller(classifier))
        label_counts = build_feature_files(base_directory, new_directory, data_loader, num_episodes,
                                           negative_example_keep_prob)
        print(label_counts)
    paths = frame.feature_file_paths("labels/RoadRunner/b1/3")
    assert len(paths) >= num_episodes, "assert len(paths) >= num_episodes, {}, {}".format(
        len(paths), num_episodes)

    data_loader = DataLoader(hparams=TensorflowClassifierHparams(**common_hparams))
    datasets = data_loader.split_episodes(paths, 1, 1, 0, use_all=True, seed=42)

    args = parser.parse_args()
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
            keep_prob=0.5, ),
        dict(
            convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6,
            fully_connected_stack_args=[20, 20],
            keep_prob=0.5,
            label_smoothing=True),
    ]

    run_experiments(
        logdir,
        data_loader,
        datasets,
        common_hparams,
        hparams_list,
        steps=20000,
        log_every=100,
        predict_episodes=False)
