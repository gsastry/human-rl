"""Incomplete, kept only for reference"""
import os
import shutil
import sys
import traceback

import numpy as np

import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

from humanrl import frame  # isort:skip
from humanrl.classifier_tf import TensorflowClassifierHparams  # isort:skip
from humanrl.classifier_tf import DataLoader, SavedClassifierTensorflow  # isort:skip


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


def convert_episode_to_tf_records(base_directory, new_directory, dataloader, path):
    episode = frame.load_episode(path)
    features, labels = dataloader.load_features_and_labels_episode(episode)
    assert path.rfind(base_directory) > -1
    new_path = path[path.rfind(base_directory) + len(base_directory) + 1:]
    new_path = os.path.splitext(new_path)[0]
    new_path = os.path.splitext(new_path)[0]
    new_path = os.path.join(new_directory, new_path + ".tfrecord")
    options = tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP)
    os.makedirs(new_path, exist_ok=True)
    for i, f in enumerate(episode.frames):
        writer = tf.python_io.TFRecordWriter(
            os.path.join(new_path, "{}.tfrecord".format(i)), options=options)
        example = tf.train.Example(features=tf.train.Features(feature={
            'action': _int64_feature([f.action]),
            'label': _int64_feature([f.label] if f.label is not None else []),
            'observation': _float_feature(f.observation.reshape(-1)),
            'observation_shape': _int64_feature(f.observation.shape),
            'image': _bytes_feature([f.image.tobytes()]),
            'image_shape': _int64_feature(f.image.shape),
        }))
        writer.write(example.SerializeToString())
        writer.close()
    return new_path


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
