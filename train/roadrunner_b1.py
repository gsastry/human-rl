import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

if __name__ == "__main__":
    import argparse

    from humanrl.classifier_tf import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")

    common_hparams = dict(
        use_action=True,
        batch_size=1000,
        input_processes=4,
        image_shape=[210, 160, 3],
        expected_positive_weight=0.2, )
    paths = frame.feature_file_paths("labels/RoadRunner/b1/0")
    data_loader = DataLoader(
        BlockerLabeller(
            info_entry="frame/classifier_label", only_catastrophe_starts=True),
        hparams=TensorflowClassifierHparams(**common_hparams))
    datasets = data_loader.split_episodes(paths, 1, 1, 1, use_all=True, seed=42)

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
            keep_prob=0.5,
            positive_weight_target=0.5),
        dict(
            convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6,
            fully_connected_stack_args=[20, 20],
            positive_weight_target=0.5),
    ]
    h = hparams_list[0]
    h.update(common_hparams)
    print(repr(TensorflowClassifierHparams(**h)))

    run_experiments(
        logdir,
        data_loader,
        datasets,
        common_hparams,
        hparams_list,
        steps=5000,
        log_every=100,
        predict_episodes=False)
