import argparse
import gzip
import itertools
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"humanrl"))

from humanrl import frame # isort:skip
from humanrl import pong_catastrophe # isort:skip
from humanrl.fd_redirector import STDERR, FDRedirector # isort:skip


parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-f', '--frames-dir', type=str, default="/tmp/pong/frames",
                    help="Directory to read and write frames to")



class TensorflowClassifier(object):

    def __init__(self):
        self.features, self.labels = self.inputs()
        self.prediction, self.loss, self.train_op = self.model(self.features, self.labels)
        self.threshold = 0.0
        tf.add_to_collection("threshold", self.threshold)

    def inputs(self):
        pass

    def model(self, features, labels):
        pass

    def threshold_from_data(self, X, y):
        y_pred = self.predict_proba(X)
        return np.min(y_pred[y])

    def metrics(self, X, y):
        metrics = {}
        y_pred = self.predict_proba(X)
        metrics['threshold'] = self.threshold_from_data(X, y)
        denom = np.count_nonzero(y == False)
        num = np.count_nonzero(np.logical_and(y == False, y_pred >= self.threshold))
        metrics['fpr'] = float(num) / float(denom)
        y_pred_bool = y_pred >= self.threshold
        if (any(y_pred_bool) and not all(y_pred_bool)):
            metrics['precision'] = precision_score(np.array(y, dtype=np.float32), y_pred_bool)
            metrics['recall'] = recall_score(y, y_pred_bool)
        return metrics

    def print_metrics(self, X, y, label):
        metrics = self.metrics(X, y)
        print("{}: {}".format(label, " ".join("{}:{:.3g}".format(key, value) for key, value in metrics.items())))

    def fit(self, X_train, y_train, X_valid, y_valid, X_test, y_test, steps=400):
        tf.global_variables_initializer().run()
        redirect=FDRedirector(STDERR)
        for i in range(steps):
            redirect.start()
            feed_dict = {self.labels:y_train}
            for key, tensor in self.features.items():
                feed_dict[tensor] = X_train[key]
            predictions, loss = sess.run([self.prediction, self.train_op], feed_dict=feed_dict)
            if i % 10 == 0:
                print("step:{} loss:{:.3g} np.std(predictions):{:.3g}".format(i, loss, np.std(predictions)))
                self.threshold = float(min(self.threshold_from_data(X_valid, y_valid), self.threshold_from_data(X_train, y_train)))
                tf.get_collection_ref("threshold")[0] = self.threshold
                self.print_metrics(X_train, y_train, "Training")
                self.print_metrics(X_valid, y_valid, "Validation")
            errors = redirect.stop()
            if errors:
                print(errors)
        self.print_metrics(X_test, y_test, "Test")


    def load_and_fit(self, episode_paths, n_train, n_valid, n_test, steps=400):
        self.fit(*self.load_data_for_training(episode_paths, n_train, n_valid, n_test), steps=steps)

    def load_data_for_training(self, episode_paths, n_train, n_valid, n_test):
        np.random.shuffle(episode_paths)
        assert n_train + n_test + n_valid < len(episode_paths)
        X_train, y_train = self.load_features(episode_paths[:n_train])
        X_test, y_test = self.load_features(episode_paths[n_train:n_train + n_test])
        X_valid, y_valid = self.load_features(episode_paths[n_train + n_test: n_train + n_test + n_valid])
        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def load_features(self, episode_paths):
        pass

    def predict(self, X):
        pass

    def save(self, checkpoint_name):
        #%mkdir -p {checkpoint_name}
        saver = tf.train.Saver()
        save_path = saver.save(tf.get_default_session(), checkpoint_name)

    def load_features(self, episode_paths):
        features = {}
        labels = None
        for episode_path in episode_paths:
            episode_features, episode_labels = self.load_features_episode(episode_path)
            for key, value in episode_features.items():
                if key not in features:
                    features[key] = value
                else:
                    features[key] = np.concatenate([features[key], value], axis=0)
            if labels is None:
                labels = episode_labels
            else:
                labels = np.concatenate([labels, episode_labels], axis=0)
            print(episode_path)
        return features, labels

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) > threshold
        precision, recall = self.metrics(X_test, y_test)

    def predict_proba(self, X):
        feed_dict = {}
        for key, tensor in self.features.items():
            feed_dict[tensor] = X[key]
        return np.reshape(self.prediction.eval(feed_dict=feed_dict),[-1])


class CatastropheClassifierTensorflow(TensorflowClassifier):
    OBSERVATION_SHAPE = [42, 42]

    def __init__(self):
        super().__init__()

    def inputs(self):
        features = {"observation": tf.placeholder(tf.float32, [None] + self.OBSERVATION_SHAPE + [1], name="observation")}
        labels = tf.placeholder(tf.float32, [None])
        return features, labels

    def model(self, features, labels):
        x = features["observation"]
        x = tf.contrib.layers.convolution2d(x, 2, kernel_size=[3, 3], stride=[2, 2], activation_fn=tf.nn.elu)
        x = tf.contrib.layers.convolution2d(x, 2, kernel_size=[3, 3], stride=[2, 2], activation_fn=tf.nn.elu)
        x = tf.contrib.layers.flatten(x)
        x = tf.contrib.layers.fully_connected(x, 100, activation_fn=tf.nn.elu)
        x = tf.contrib.layers.fully_connected(x, 100, activation_fn=tf.nn.elu)
        logits = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        prediction = tf.sigmoid(logits)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.expand_dims(labels, axis=1)))
        train_op = tf.contrib.layers.optimize_loss(
          loss, tf.contrib.framework.get_global_step(), optimizer='Adam',
          learning_rate=0.01)
        tf.add_to_collection('prediction', prediction)
        tf.add_to_collection('loss', loss)
        return prediction, loss, train_op

    def load_features_episode(self, episode_path):
        features = {}
        episode = frame.load_episode(episode_path)
        observations = [frame.observation for frame in episode.frames]
        actions = [frame.action for frame in episode.frames]
        features['observation'] = np.concatenate([np.expand_dims(observation, axis=0) for observation in observations], axis=0)
        labels = np.array([pong_catastrophe.is_catastrophe(observation, location="bottom") for observation in observations])
        return features, labels

class CatastropheBlockerTensorflow(TensorflowClassifier):
    OBSERVATION_SHAPE = [42, 42]

    def __init__(self, learning_rate=0.002, block_radius=0):
        self.learning_rate = learning_rate
        self.block_radius = block_radius
        super().__init__()

    def inputs(self):
        features = {}
        features["observation"] = tf.placeholder(tf.float32, [None] + self.OBSERVATION_SHAPE + [1], name="observation")
        features["action"] = tf.placeholder(tf.int32, [None, 1], name="action")
        for key, value in features.items():
            tf.add_to_collection("features", value)
        labels = tf.placeholder(tf.float32, [None])
        return features, labels

    def model(self, features, labels):
        x = features["observation"]
        x = tf.contrib.layers.convolution2d(x, 2, kernel_size=[3, 3], stride=[2, 2], activation_fn=tf.nn.elu)
        x = tf.contrib.layers.convolution2d(x, 2, kernel_size=[3, 3], stride=[2, 2], activation_fn=tf.nn.elu)
        actions = tf.one_hot(tf.reshape(features["action"],[-1]), depth=6, on_value=1.0, off_value=0.0, axis=1)
        x = tf.concat(1, [tf.contrib.layers.flatten(x),  actions])
        x = tf.contrib.layers.fully_connected(x, 100, activation_fn=tf.nn.elu)
        x = tf.contrib.layers.fully_connected(x, 100, activation_fn=tf.nn.elu)
        logits = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)
        prediction = tf.sigmoid(logits, name="prediction")
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.expand_dims(labels, axis=1)),name="loss")
        train_op = tf.contrib.layers.optimize_loss(
          loss, tf.contrib.framework.get_global_step(), optimizer='Adam',
          learning_rate=self.learning_rate)
        tf.add_to_collection('prediction', prediction)
        tf.add_to_collection('loss', loss)
        return prediction, loss, train_op

    def load_features_episode(self, episode_path):
        features = {}
        episode = frame.load_episode(episode_path)


        observations = [frame.observation for frame in episode.frames]
        actions = [frame.action for frame in episode.frames]
        features['observation'] = np.concatenate([np.expand_dims(observation, axis=0) for observation in observations], axis=0)
        features['action'] = np.expand_dims(np.array(actions), axis=1)

        is_catastrophe = np.array([pong_catastrophe.is_catastrophe(observation, location="bottom") for observation in observations])
        is_catastrophe = is_catastrophe[1:]
        for key, value in features.items():
            features[key] = features[key][:-1]

        labels = is_catastrophe
        for i in range(1, self.block_radius):
            labels = np.logical_or(labels, (np.pad(is_catastrophe[i:], (0,i), 'constant')))

        # Filter out instances where a catastrophe is already in progress
        not_already_catastrophe = np.logical_not(is_catastrophe[0:-1])
        labels = labels[1:][not_already_catastrophe]
        for key, value in features.items():
            features[key] = features[key][1:][not_already_catastrophe]


        ### USE HUMAN LABELS
        new_labels = np.array( [bool(frame.label) for frame in episode.frames] )
        new_labels = new_labels[:len(labels)]
        labels = new_labels

        print( [frame.label for frame in episode.frames] )
        print( 'total cats', np.sum(labels) )

        return features, labels



def display_example(x,i):
    # imsize=42*42
    # observation = x[2:imsize+2].reshape([42,42])
    # observation2 = x[imsize+2:].reshape([42,42])
    # print(observation.shape)
    # Plot the grid
    x = x.reshape(42,42)
    plt.imshow(x)
    plt.gray()
    #plt.show()
    plt.savefig('/tmp/catastrophe/frame_{}.png'.format(i))



if __name__ == "__main__":
    args = parser.parse_args()


    episode_paths = frame.episode_paths( args.frames_dir )
#     g = tf.Graph()
#     with g.as_default():
#         classifier = CatastropheClassifierTensorflow()
#         data = classifier.load_data_for_training(episode_paths, 1, 1, 1)
#         sess = tf.Session(graph=g)
#         with sess.as_default():
#             classifier.fit(*data, steps=10)
# #             %mkdir -p "/tmp/foo/0.ckpt"
#             classifier.save(checkpoint_name="/tmp/foo/classifier/0.ckpt")
    print(episode_paths)

    g = tf.Graph()
    with g.as_default():
        blocker = CatastropheBlockerTensorflow(block_radius=1)
        data = blocker.load_data_for_training(episode_paths, 1, 1, 1)
        X_train, y_train, X_valid, y_valid, X_test, y_test = data

        number_frames = 700



        for i in range(number_frames):
            x = X_train['observation'][i]
            if y_train[i]:
                display_example(x,i)

        #print( y_train[:number_frames] )


        #sess = tf.Session(graph=g)
        #with sess.as_default():
        #    blocker.fit(*data, steps=10)
        #    blocker.save(checkpoint_name="/tmp/foo/blocker/0.ckpt")
