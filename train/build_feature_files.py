import argparse
import os
import pathlib
import pickle
import shutil
import sys
import traceback
import unittest
from collections import defaultdict

import numpy as np
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

import frame  # isort:skip
from classifier_tf import BlockerLabeller, TensorflowClassifierHparams  # isort:skip
from classifier_tf import DataLoader, SavedCatastropheClassifierTensorflow  # isort:skip

parser = argparse.ArgumentParser(
    description="""Turns a set of episodes into a set of feature files, which contain raw
features and labels for a subset of an episode, possibly randomly selected.""")
parser.add_argument(
    '-f',
    '--frames-dir',
    type=str,
    default="/tmp/pong/frames",
    help="Directory to read and write frames to")
parser.add_argument(
    '--env-id', type=str, default="PongDeterministic-v3", help="ID of environment to run")
parser.add_argument(
    '-n', '--num-episodes', type=int, default=100, help="Number of episodes to generate")


def path_relative_to_new_directory(base_directory, new_directory, path, new_ext):
    assert path.rfind(base_directory) > -1
    s = path[path.rfind(base_directory) + len(base_directory) + 1:]
    suffix = "".join(pathlib.Path(s).suffixes)
    s = s[:-(len(suffix))]
    s = os.path.join(new_directory, s + new_ext)
    return s


class Test_path_relative_to_new_directory(unittest.TestCase):
    def test_1(self):
        self.assertEqual(
            path_relative_to_new_directory("logs/Pong", "labels/Pong",
                                           "logs/Pong/episodes/w0/e1.pkl", ".frames"),
            "labels/Pong/episodes/w0/e1.frames")
        self.assertEqual(
            path_relative_to_new_directory("logs/Pong", "labels/Pong",
                                           "logs/Pong/episodes/w0/e1.pkl.gz", ".frames"),
            "labels/Pong/episodes/w0/e1.frames")
        self.assertEqual(
            path_relative_to_new_directory("logs/Pong", "labels/Pong",
                                           "base/logs/Pong/episodes/w0/e1.pkl.gz", ".frames"),
            "labels/Pong/episodes/w0/e1.frames")


def build_feature_files(base_directory,
                        new_directory,
                        data_loader,
                        n=None,
                        negative_example_keep_prob=1.0):
    os.makedirs(new_directory, exist_ok=False)
    episode_paths = frame.episode_paths(base_directory)
    label_counts = [0, 0]
    if n is not None:
        np.random.shuffle(episode_paths)
        episode_paths = episode_paths[:n]
    for episode_path in tqdm.tqdm(episode_paths):
        try:
            features, labels = data_loader.load_features_and_labels([episode_path])
        except:
            traceback.print_exc()
        else:
            keep = np.logical_or(labels, (np.less(
                np.random.rand(len(labels)), negative_example_keep_prob)))
            labels = labels[keep]

            for i in range(len(label_counts)):
                label_counts[i] += np.count_nonzero(labels == i)
            features = {k: v[keep] for k, v in features.items()}
            new_path = path_relative_to_new_directory(base_directory, new_directory, episode_path,
                                                      ".features")
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with open(new_path, 'wb') as f:
                pickle.dump((features, labels), f)
    return label_counts


def action_label_counts(directory, data_loader, n_actions=18, n=None):
    episode_paths = frame.episode_paths(directory)
    label_counts = [0, 0]
    action_label_counts = [[0, 0] for i in range(n_actions)]
    if n is not None:
        np.random.shuffle(episode_paths)
        episode_paths = episode_paths[:n]
    for episode_path in tqdm.tqdm(episode_paths):
        try:
            features, labels = data_loader.load_features_and_labels([episode_path])
        except:
            traceback.print_exc()
        else:
            for label in range(len(label_counts)):
                label_counts[label] += np.count_nonzero(labels == label)
                for action in range(n_actions):
                    actions = np.reshape(np.array(features["action"]), [-1])
                    action_label_counts[action][label] += np.count_nonzero(
                        np.logical_and(labels == label, actions == action))
    return label_counts, action_label_counts


if __name__ == "__main__":
    tp = unittest.main(exit=False)
    if not tp.result.wasSuccessful():
        sys.exit(False)

    base_directory = "logs/RoadRunner"
    new_directory = "labels/RoadRunner/b1/4"
    if os.path.exists(new_directory):
        shutil.rmtree(new_directory)
    classifier = SavedCatastropheClassifierTensorflow("models/roadrunner/c1/classifier/final.ckpt")
    data_loader = DataLoader(
        hparams=classifier.classifier.hparams, labeller=BlockerLabeller(classifier))
    label_counts = build_feature_files(
        base_directory, new_directory, data_loader, n=10, negative_example_keep_prob=0.5)
    print(label_counts)
    L = frame.feature_file_paths(new_directory)
    data_loader.load_features_and_labels(L[:1])
