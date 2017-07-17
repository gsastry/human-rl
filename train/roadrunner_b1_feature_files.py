import os
import sys

from build_feature_files import build_feature_files

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

if __name__ == "__main__":
    import argparse

    from humanrl import frame
    from humanrl.classifier_tf import (SavedCatastropheClassifierTensorflow,
                                                      DataLoader, BlockerLabeller)

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default="")

    common_hparams = dict(use_action=True, image_shape=[210, 160, 3])

    base_directory = "logs/RoadRunner"
    new_directory = "labels/RoadRunner/b1/0"
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
    paths = frame.feature_file_paths("labels/RoadRunner/b1/0")
