import os
import shutil
import sys
import traceback

import numpy as np

import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

import frame  # isort:skip
from classifier_tf import TensorflowClassifierHparams  # isort:skip
from classifier_tf import DataLoader, SavedClassifierTensorflow  # isort:skip


def copy_episodes(indir, outdir, n):
    episode_paths = frame.episode_paths(indir)
    np.random.shuffle(episode_paths)
    episode_paths = episode_paths[:n]
    start = len(indir)
    for p in tqdm.tqdm(episode_paths):
        assert p.startswith(indir), p
        outfile = outdir + p[start:]
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        shutil.copyfile(p, outfile)


def label_episodes(directory, classifier):
    episode_paths = frame.episode_paths(directory)
    data_loader = DataLoader(hparams=classifier.hparams)
    for episode_path in tqdm.tqdm(episode_paths):
        try:
            data_loader.predict_episodes(classifier, [episode_path], prefix="frame/classifier_")
        except EOFError as e:
            traceback.print_exception(e)
            print("Error reading {}".format(episode_path))
            os.remove(episode_path)


if __name__ == "__main__":
    outdir = "labels/RoadRunner/b1/2"
    copy_episodes("logs/RoadRunner", outdir, 100)
    classifier = SavedClassifierTensorflow("models/roadrunner/c1/classifier/final.ckpt")
    label_episodes(outdir, classifier)
