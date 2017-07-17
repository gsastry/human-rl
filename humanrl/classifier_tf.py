import go_vncdriver  # isort:skip

import ast
import hashlib
import math
import os
import pathlib
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import defaultdict, namedtuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import cv2
import frame
import input_thread
from classifier import *

# logger = logging.getLogger('tensorflow').handlers.clear()
# logging.getLogger("gym.core").handlers = []

MistakeTuple = namedtuple('MistakeTuple', ['y_pred', 'features'])


class ScoreDistribution:
    def __init__(self):
        self.threshold = None
        self.y_train = None
        self.y_pred_train = None
        self.y_valid = None
        self.y_pred_valid = None
        self.y_test = None
        self.y_pred_test = None


def score_distribution_for_classifier(classifier,
                                      dataloader,
                                      episode_paths_train=tuple(),
                                      episode_paths_valid=tuple(),
                                      episode_paths_test=tuple()):
    sd = ScoreDistribution()
    if episode_paths_train:
        sd.y_train, sd.y_pred_train = labels_and_predictions(classifier, dataloader,
                                                             episode_paths_train)
    if episode_paths_valid:
        sd.y_valid, sd.y_pred_valid = labels_and_predictions(classifier, dataloader,
                                                             episode_paths_valid)
    if episode_paths_test:
        sd.y_test, sd.y_pred_test = labels_and_predictions(classifier, dataloader,
                                                           episode_paths_test)
    return sd


def labels_and_predictions(classifier, dataloader, episode_paths):
    y_all = []
    y_pred_all = []
    for ep in episode_paths:
        X, y = dataloader.load_features_and_labels([ep])
        y_pred = classifier.predict_proba(X)
        del X
        y_all.append(y)
        y_pred_all.append(y_pred)
    y = np.concatenate(y_all)
    y_pred = np.concatenate(y_pred_all)
    return y, y_pred


def threshold_from_predictions(y, y_pred, false_positive_margin=0, recall=1):
    """Determines a threshold for classifying examples as positive

    Args:
        y: labels
        y_pred: scores from the classifier
        recall: Threshold is set to classify at least this fraction of positive
            labelled examples as positive
        false_positive_margin: Threshold is set to acheive desired recall, and
            then is extended to include an additional fraction of negative
            labelled examples equal to false_positive_margin (This allows adding
            a buffer to the threshold while maintaining a constant "cost")
    """
    n_positive = np.count_nonzero(y)

    n_negative = len(y) - n_positive
    if n_positive == 0:
        return np.max(y_pred)
    if false_positive_margin == 0 and recall == 1:
        return np.min(y_pred[y])
    ind = np.argsort(y_pred)
    y_pred_sorted = y_pred[ind]
    y_sorted = y[ind]
    so_far = [0, 0]
    j = 0
    for i in reversed(range(len(y_sorted))):
        so_far[y_sorted[i]] += 1
        if so_far[1] >= int(np.floor(recall * n_positive)):
            j = i
            break
    so_far = [0, 0]
    if false_positive_margin == 0:
        return y_pred_sorted[j]
    k = 0
    for i in reversed(range(j)):
        so_far[y_sorted[i]] += 1
        if so_far[0] >= false_positive_margin * n_negative:
            k = i
            break
    return y_pred_sorted[k]


class Test_threshold_from_predictions(unittest.TestCase):
    def test_false_positive_margin(self):
        y_pred = np.arange(10, -1, -1) * 0.1
        y = np.array([False] * 5 + [True] + [False] * 5)
        self.assertEqual(len(y_pred), len(y))
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred, 0), 0.5)
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred, 0.1), 0.4)
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred, 0.4), 0.1)
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred, 0.6), 0.0)

    def test_recall(self):
        y_pred = np.arange(10, 0, -1) * 0.1
        y = np.array([True] * 10)
        self.assertEqual(len(y_pred), len(y))
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred), 0.1)
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred, recall=0.5), 0.6)
        self.assertAlmostEqual(threshold_from_predictions(y, y_pred, recall=0.1), 1.0)


def classification_metrics(y, y_pred, threshold):
    metrics = {}
    metrics['threshold'] = threshold_from_predictions(y, y_pred, 0)
    metrics['np.std(y_pred)'] = np.std(y_pred)
    metrics['positive_frac_batch'] = float(np.count_nonzero(y == True)) / len(y)
    denom = np.count_nonzero(y == False)
    num = np.count_nonzero(np.logical_and(y == False, y_pred >= threshold))
    if denom > 0:
        metrics['fpr'] = float(num) / float(denom)
    if any(y) and not all(y):
        metrics['auc'] = roc_auc_score(y, y_pred)
        y_pred_bool = y_pred >= threshold
        if (any(y_pred_bool) and not all(y_pred_bool)):
            metrics['precision'] = precision_score(np.array(y, dtype=np.float32), y_pred_bool)
            metrics['recall'] = recall_score(y, y_pred_bool)
    return metrics


class TensorflowClassifier:
    def __init__(self, dataloader, hparams):
        self.hparams = hparams
        self.dataloader = dataloader
        self.build_inputs()
        self.build_model()
        self.threshold = 0.0
        self.summary_tensors = {}

    def build_inputs(self):
        self.features = {}
        if self.hparams.use_observation:
            self.features["observation"] = tf.placeholder(
                tf.float32, [None] + self.hparams.observation_shape, name="observation")
        if self.hparams.use_action:
            self.features["action"] = tf.placeholder(tf.int32, [None, 1], name="action")
        if self.hparams.use_image:
            self.features["image"] = tf.placeholder(
                tf.float32, [None] + self.hparams.image_shape, name="image")
        self.labels = tf.placeholder(tf.float32, [None])

    def write_summaries(self, X, y, label, step, summary_writer=None):
        if not X:
            return
        y_pred, loss = self.predict_proba_with_loss(X, y)
        metrics = classification_metrics(y, y_pred, self.threshold)
        metrics['loss'] = loss
        if summary_writer is not None:
            summary = tf.Summary()
            for key, value in metrics.items():
                summary.value.add(tag="metrics/{}".format(key), simple_value=float(value))
            if not self.summary_tensors:
                self.summary_tensors["positive_predictions_input"] = tf.placeholder(
                    tf.float32, [None], "positive_predictions_input")
                self.summary_tensors["positive_predictions"] = tf.summary.histogram(
                    "positive_predictions", self.summary_tensors["positive_predictions_input"])
                self.summary_tensors["negative_predictions_input"] = tf.placeholder(
                    tf.float32, [None], "negative_predictions_input")
                self.summary_tensors["negative_predictions"] = tf.summary.histogram(
                    "negative_predictions", self.summary_tensors["negative_predictions_input"])
            summary_writer.add_summary(
                self.summary_tensors["positive_predictions"].eval(
                    feed_dict={self.summary_tensors["positive_predictions_input"]: y_pred[y]}),
                step)
            summary_writer.add_summary(
                self.summary_tensors["negative_predictions"].eval(
                    feed_dict={self.summary_tensors["negative_predictions_input"]: y_pred[~y]}),
                step)
            summary_writer.add_summary(summary, step)
            summary_writer.flush()

    def launch_queues(self, episode_paths):
        def local_queue_thread(input_queue, output_queue):
            while (True):
                x = input_queue.get()
                output_queue.put(x)

        ctx = input_thread.ctx
        multiprocess_queue = ctx.Queue(maxsize=self.hparams.input_process_queue_size)

        input_processes = [
            ctx.Process(
                target=input_thread.input_thread_fn,
                daemon=True,
                args=(multiprocess_queue, self.dataloader, self.hparams.batch_size, episode_paths))
            for i in range(self.hparams.input_processes)
        ]

        thread_queue = queue.Queue(maxsize=self.hparams.read_input_queue_size)

        transfer_threads = [
            threading.Thread(
                target=local_queue_thread, daemon=True, args=(multiprocess_queue, thread_queue))
            for i in range(self.hparams.read_input_threads)
        ]
        for thread in transfer_threads:
            thread.start()

        for input_process in input_processes:
            input_process.start()

        return thread_queue, input_processes

    def log_mistakes(self, logdir, step, X, y):
        y_pred = self.predict_proba(X)
        false_negative_loss = y * (1.0 - y_pred)
        false_positive_loss = (1 - y) * y_pred
        false_negative_ind = np.argmax(false_negative_loss)
        false_positive_ind = np.argmax(false_positive_loss)
        fname = os.path.join(logdir, "Training", "mistakes", "fn{}.pkl".format(step))
        with open(fname, 'wb') as f:
            features = {name: value[false_negative_ind] for name, value in X.items()}
            pickle.dump(MistakeTuple(y_pred[false_negative_ind], features), f)
        fname = os.path.join(logdir, "Training", "mistakes", "fp{}.pkl".format(step))
        with open(fname, 'wb') as f:
            features = {name: value[false_positive_ind] for name, value in X.items()}
            pickle.dump(MistakeTuple(y_pred[false_positive_ind], features), f)

    def fit(self,
            episode_paths_train,
            episode_paths_valid,
            episode_paths_test,
            steps=400,
            log_every=100,
            logdir=None,
            sess=None,
            X_extra=None,
            y_extra=None,
            max_checkpoints=10,
            max_mistakes=100):
        max_checkpoints = max(min(max_checkpoints, steps / 100), 2)

        last_checkpoint_step = 0
        last_mistake_step = 0
        start_time = time.time()
        if sess is None:
            sess = tf.get_default_session()
        training_summary_writer, validation_summary_writer, extra_summary_writer = None, None, None
        if logdir is not None:
            training_summary_writer = tf.summary.FileWriter(
                os.path.join(logdir, "Training"), graph=tf.get_default_graph())
            validation_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "Validation"))
            if X_extra is not None:
                extra_summary_writer = tf.summary.FileWriter(os.path.join(logdir, "Extra"))
            os.makedirs(os.path.join(logdir, "Training", "mistakes"), exist_ok=True)
        tf.global_variables_initializer().run()

        timing_last_i = 0
        timing = defaultdict(lambda: 0.0)

        thread_queue_train, input_processes_train = None, []
        if self.hparams.multiprocess:
            thread_queue_train, input_processes_train = self.launch_queues(episode_paths_train)

        try:
            for i in range(steps):
                t1 = time.time()
                if self.hparams.multiprocess:
                    X_train, y_train = thread_queue_train.get()
                else:
                    eps = np.random.choice(episode_paths_valid, size=1)
                    X_train, y_train = self.dataloader.load_features_and_labels(eps)

                feed_dict = {self.is_training: True, self.labels: y_train}
                for key, tensor in self.features.items():
                    feed_dict[tensor] = X_train[key]
                t2 = time.time()
                predictions, loss = sess.run([self.prediction, self.train_op], feed_dict=feed_dict)
                t3 = time.time()
                timing["data_load_step_time"] += t2 - t1
                timing["train_step_time"] += t3 - t2
                timing["total_step_time"] += t3 - t1

                if i % log_every == 0 or i == steps - 1:
                    if self.hparams.verbose:
                        print('Step {} of {}'.format(i, steps))
                    eps = np.random.choice(
                        episode_paths_valid,
                        size=min(self.hparams.valid_num_episodes, len(episode_paths_valid)),
                        replace=False)
                    X_valid, y_valid = self.dataloader.load_features_and_labels(eps)
                    y_pred_train = self.predict_proba(X_train)
                    self.threshold = threshold_from_predictions(y_train, y_pred_train,
                                                                self.hparams.false_positive_margin)
                    self.write_summaries(X_train, y_train, "Training", i, training_summary_writer)
                    self.write_summaries(X_valid, y_valid, "Validation", i,
                                         validation_summary_writer)
                    if X_extra is not None:
                        self.write_summaries(X_extra, y_extra, "Extra", i, extra_summary_writer)

                    if logdir is not None and (i - last_mistake_step >= (steps / max_mistakes)):
                        last_mistake_step = i
                        self.log_mistakes(logdir, i, X_train, y_train)

                    if training_summary_writer is not None:
                        summary = tf.Summary()
                        for key, value in timing.items():
                            summary.value.add(
                                tag="timing/{}".format(key),
                                simple_value=float(value) / (i - timing_last_i + 1))
                        training_summary_writer.add_summary(summary, i)
                        training_summary_writer.flush()
                    timing_last_i = i
                    timing = defaultdict(lambda: 0.0)

                    if logdir is not None and (i - last_checkpoint_step >=
                                               (steps / max_checkpoints)):
                        last_checkpoint_step = i
                        self.save_checkpoint(os.path.join(logdir, "{}.ckpt".format(i)))
        finally:
            for input_process in input_processes_train:
                input_process.terminate()

        if self.hparams.verbose:
            print('Computing predictions for threshold')
        sd = score_distribution_for_classifier(self, self.dataloader, episode_paths_train,
                                               episode_paths_valid, episode_paths_test)

        # Set threshold to the minimum threshold between the training set and
        # the validation set
        self.threshold = min(
            threshold_from_predictions(sd.y_train, sd.y_pred_train,
                                       self.hparams.false_positive_margin),
            threshold_from_predictions(sd.y_valid, sd.y_pred_valid,
                                       self.hparams.false_positive_margin))
        sd.threshold = self.threshold

        with open(os.path.join(logdir, "score_distribution.pkl"), "wb") as f:
            pickle.dump(sd, f)
        self.save_checkpoint(os.path.join(logdir, "final.ckpt"))

        print("Elapsed time: {}s".format(time.time() - start_time))
        print("Prediction time {}s".format(self.prediction_time(X_train)))

    def labels_and_predictions(self, episode_paths):
        return labels_and_predictions(self, self.dataloader, episode_paths)

    def prediction_time(self, X):
        start_time = time.time()
        dp = {}
        for key, value in X.items():
            dp[key] = value[np.newaxis, 0]
        times = 100
        for _ in range(times):
            self.predict_proba(dp)
        return (time.time() - start_time) / times

    def save_checkpoint(self, checkpoint_name):
        tf.get_collection_ref("threshold")[:] = [float(self.threshold)]
        tf.get_collection_ref("features")[:] = self.features.values()
        tf.get_collection_ref("loss")[:] = [self.loss]
        tf.get_collection_ref("prediction")[:] = [self.prediction]

        os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), checkpoint_name)

        with open(os.path.join(os.path.dirname(checkpoint_name), "hparams.txt"), "w") as f:
            f.write(repr(self.hparams.__dict__))

    def predict(self, X):
        return self.apply_threshold(self.predict_proba(X))

    def predict_proba(self, X):
        feed_dict = {}
        for key, tensor in self.features.items():
            feed_dict[tensor] = X[key]
        return np.reshape(self.prediction.eval(feed_dict=feed_dict), [-1])

    def predict_proba_with_loss(self, X, y):
        feed_dict = {}
        feed_dict[self.labels] = y
        for key, tensor in self.features.items():
            feed_dict[tensor] = X[key]
        prediction, loss = tf.get_default_session().run(
            [self.prediction, self.loss], feed_dict=feed_dict)
        return np.reshape(prediction, [-1]), loss

    def build_model(self):
        self.is_training = tf.constant(False)
        keep_prob = tf.cond(self.is_training, lambda: tf.constant(self.hparams.keep_prob),
                            lambda: tf.constant(1.0))

        activation_fn = lambda x: tf.nn.dropout(tf.nn.elu(x), keep_prob)
        normalizer_fn = tf.contrib.layers.batch_norm if self.hparams.batch_normalization else None
        collections = {"weights": [tf.GraphKeys.WEIGHTS]}

        fully_connected_input_features = []

        conv_input = None
        if self.hparams.use_observation:
            conv_input = self.features["observation"]
        elif self.hparams.use_image:
            conv_input = self.features["image"]
        if conv_input is not None:
            x = tf.contrib.layers.stack(
                conv_input,
                tf.contrib.layers.convolution2d,
                self.hparams.convolution2d_stack_args,
                activation_fn=activation_fn,
                variables_collections=collections,
                normalizer_fn=normalizer_fn)
            print("Convolution Output Shape: {}".format(x.get_shape()))
            x = tf.contrib.layers.flatten(x)
            fully_connected_input_features.append(x)

        if self.hparams.use_action:
            action = tf.one_hot(
                tf.reshape(self.features["action"], [-1]),
                depth=self.hparams.action_depth,
                on_value=1.0,
                off_value=0.0,
                axis=1,
                name="action_one_hot")
            fully_connected_input_features.append(action)

        x = tf.concat(fully_connected_input_features, 1)
        x = tf.contrib.layers.stack(
            x,
            tf.contrib.layers.fully_connected,
            self.hparams.fully_connected_stack_args,
            activation_fn=activation_fn,
            variables_collections=collections,
            normalizer_fn=normalizer_fn)
        logits = tf.contrib.layers.fully_connected(x, 1, activation_fn=None)

        # if self.hparams.clip_logits is not None:
        #     logits = tf.clip_by_value(logits, *self.hparams.clip_logits)
        self.prediction = tf.sigmoid(logits, name="prediction")
        labels = tf.expand_dims(self.labels, axis=1)
        if self.hparams.label_smoothing:
            labels = 0.9 * labels + 0.1 * (1 - labels)
        positive_multiplier = 1.0
        negative_multiplier = 1.0
        if self.hparams.positive_weight_target is not None:
            # batch_positive_weight = tf.reduce_mean(tf.to_float(labels))
            positive_multiplier = self.hparams.positive_weight_target / self.hparams.expected_positive_weight
            negative_multiplier = (1 - self.hparams.expected_positive_weight) / (
                1 - self.hparams.positive_weight_target)
        # positive_loss = tf.reduce_mean(labels * -tf.log(tf.sigmoid(logits) + 1e-9))
        # negative_loss = tf.reduce_mean((1 - labels) * -tf.log(1 - tf.sigmoid(logits) + 1e-9))
        # self.loss = tf.add(positive_multiplier * positive_loss,
        #                    negative_multiplier * negative_loss,
        #                    name="loss")
        self.loss = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=logits, targets=labels,
                pos_weight=positive_multiplier / negative_multiplier),
            name="loss") * negative_multiplier

        if self.hparams.l1_regularization_weight > 0.0:
            regularizer = tf.contrib.layers.l1_regularizer(self.hparams.l1_regularization_weight)
            self.loss += tf.contrib.layers.apply_regularization(regularizer)
        if self.hparams.l2_regularization_weight > 0.0:
            regularizer = tf.contrib.layers.l2_regularizer(self.hparams.l2_regularization_weight)
            self.loss += tf.contrib.layers.apply_regularization(regularizer)

        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adam',
            learning_rate=self.hparams.learning_rate,
            summaries=[])

        if self.hparams.check_numerics:
            self.train_op = tf.group(self.train_op, tf.add_check_numerics_ops())


class SavedClassifierTensorflow:
    def __init__(self, checkpoint_file):

        checkpoint_dir = os.path.dirname(checkpoint_file)
        hparams_file = os.path.join(checkpoint_dir, "hparams.txt")
        hparams_dict = {}
        if os.path.isfile(hparams_file):
            with open(hparams_file) as f:
                hparams_dict = ast.literal_eval(f.read())
        self.hparams = TensorflowClassifierHparams(**hparams_dict)
        self.graph = tf.Graph()
        with self.graph.as_default():
            print("loading from file {}".format(checkpoint_file))
            config = tf.ConfigProto(
                device_count={'GPU': 0}, )
            config.gpu_options.visible_device_list = ""
            self.session = tf.Session(config=config)
            new_saver = tf.train.import_meta_graph(checkpoint_file + ".meta", clear_devices=True)
            new_saver.restore(self.session, checkpoint_file)

            self.features = {}

            if self.hparams.use_image:
                self.features["image"] = self.graph.get_tensor_by_name("image:0")
            if self.hparams.use_observation:
                self.features["observation"] = self.graph.get_tensor_by_name("observation:0")
            if self.hparams.use_action:
                self.features["action"] = self.graph.get_tensor_by_name("action:0")
            self.prediction = tf.get_collection('prediction')[0]
            self.loss = tf.get_collection('loss')[0]
            self.threshold = tf.get_collection('threshold')[0]

    def make_features(self, obs=None, action=None):
        features = {}
        if obs is not None:
            if isinstance(obs, list):
                obs = [process_image(o, self.hparams) for o in obs]
                features["image"] = np.concatenate(obs, axis=0)
            else:
                obs = process_image(obs, self.hparams)
                features["image"] = obs
        if action is not None:
            if isinstance(action, list):
                features["action"] = np.array([action])
            else:
                features["action"] = np.array([[action]])
        return features

    def predict_proba_raw(self, obs=None, action=None):
        prediction = self.predict_proba(self.make_features(obs, action))
        if len(prediction) == 1:
            return prediction[0]
        return prediction

    def predict_proba(self, features):
        feed_dict = {}
        for name, tensor in self.features.items():
            feed_dict[tensor] = features[name]
        prediction = self.prediction.eval(session=self.session, feed_dict=feed_dict)
        return prediction[:, 0]

    def predict_raw(self, obs=None, action=None):
        return np.greater_equal(self.predict_proba_raw(obs, action), self.threshold)

    def predict_raw_with_score(self, obs=None, action=None):
        score = self.predict_proba_raw(obs, action)
        return self.apply_threshold(score), score

    def apply_threshold(self, y):
        return np.greater_equal(y, self.threshold)

    def predict(self, features):
        return self.apply_threshold(self.predict_proba(features))

    def __del__(self):
        self.session.close()


class SavedClassifierTensorflowEnsemble:
    def __init__(self, checkpoint_file_list, threshold=2):
        self.classifiers = [
            SavedClassifierTensorflow(checkpoint_file) for checkpoint_file in checkpoint_file_list
        ]
        self.threshold = threshold

    def predict_proba(self, features):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict_raw(features))
        return np.count_nonzero(predictions)

    def predict_proba_raw(self, obs=None, action=None):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict_raw(obs, action))
        return np.count_nonzero(predictions)

    def predict_raw(self, obs=None, action=None):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict_raw(obs, action))
        return self.apply_threshold(np.count_nonzero(predictions))

    def predict_raw_with_score(self, obs=None, action=None):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict_raw(obs, action))
        return self.apply_threshold(np.count_nonzero(predictions)), np.count_nonzero(predictions)

    def predict(self, features):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict(features))
        return self.apply_threshold(np.count_nonzero(predictions))

    def apply_threshold(self, y):
        return np.greater_equal(y, self.threshold)


class SavedCatastropheBlockerTensorflow:
    def __init__(self, checkpoint_file, n_actions, threshold=None):
        if isinstance(checkpoint_file, list):
            if len(checkpoint_file) >= 2:
                self.classifier = SavedClassifierTensorflowEnsemble(checkpoint_file)
            else:
                self.classifier = SavedClassifierTensorflow(checkpoint_file[0])
        else:
            self.classifier = SavedClassifierTensorflow(checkpoint_file)
        self.n_actions = n_actions
        if threshold is None:
            self.threshold = self.classifier.threshold
        else:
            self.threshold = threshold

    def apply_threshold(self, y):
        return np.greater_equal(y, self.threshold)

    def should_block(self, obs, action):
        return self.apply_threshold(self.classifier.predict_proba_raw(obs, action))

    def should_block_with_score(self, obs, action):
        score = self.classifier.predict_proba_raw(obs, action)
        return self.apply_threshold(score), score

    def allowed_actions_with_scores(self, obs):
        """Allow all actions that are safe or (if none are safe), the safest action"""
        action_scores = []
        for action in range(self.n_actions):
            action_scores.append(self.classifier.predict_proba_raw(obs, action))
        safe = np.logical_not(self.apply_threshold(action_scores))

        allowed_actions = [action for action in range(self.n_actions) if safe[action]]
        if not allowed_actions:
            allowed_actions = [np.argmin(action_scores)]

        return allowed_actions, action_scores

    def allowed_actions(self, obs):
        return self.allowed_actions_with_scores(obs)[0]

    def __del__(self):
        pass


class SavedCatastropheClassifierTensorflow:
    def __init__(self, checkpoint_file, threshold=None):
        self.checkpoint_file = checkpoint_file
        if isinstance(checkpoint_file, list):
            if len(checkpoint_file) >= 2:
                self.classifier = SavedClassifierTensorflowEnsemble(checkpoint_file)
            else:
                self.classifier = SavedClassifierTensorflow(checkpoint_file[0])
        else:
            self.classifier = SavedClassifierTensorflow(checkpoint_file)
        if threshold is None:
            self.threshold = self.classifier.threshold
        else:
            self.threshold = threshold

    def apply_threshold(self, y):
        return np.greater_equal(y, self.threshold)

    def is_catastrophe(self, obs):
        if obs is None:
            return False
        return self.classifier.predict_raw(obs)

    def is_catastrophe_with_score(self, obs):
        if obs is None:
            return False
        return self.classifier.predict_raw_with_score(obs)

    def __getstate__(self):
        return (self.checkpoint_file, )

    def __setstate__(self, newstate):
        self.__init__(*newstate)


class TensorflowClassifierHparams:
    def __init__(self,
                 learning_rate=0.002,
                 use_observation=False,
                 observation_shape=[42, 42, 1],
                 use_action=False,
                 action_depth=18,
                 use_image=True,
                 original_image_shape=[210, 160, 3],
                 image_shape=[105, 80, 3],
                 keep_prob=1.0,
                 l1_regularization_weight=0.0,
                 l2_regularization_weight=0.0,
                 batch_normalization=False,
                 convolution2d_stack_args=[(4, [3, 3], [2, 2])] * 5,
                 fully_connected_stack_args=[10, 10],
                 batch_size=1000,
                 test_num_episodes=1,
                 valid_num_episodes=1,
                 image_crop_region=None,
                 label_smoothing=False,
                 input_processes=4,
                 input_process_queue_size=5,
                 read_input_threads=2,
                 read_input_queue_size=2,
                 positive_weight_target=None,
                 expected_positive_weight=None,
                 clip_logits=[-10, 10],
                 check_numerics=False,
                 false_positive_margin=0.0,
                 multiprocess=True,
                 verbose=False,
                 **kwargs):
        self.learning_rate = learning_rate
        self.use_observation = use_observation
        self.observation_shape = observation_shape
        self.use_action = use_action
        self.action_depth = action_depth
        self.use_image = use_image
        self.original_image_shape = original_image_shape
        self.image_shape = image_shape
        self.keep_prob = keep_prob
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
        self.batch_normalization = batch_normalization
        self.convolution2d_stack_args = convolution2d_stack_args
        self.fully_connected_stack_args = fully_connected_stack_args
        self.batch_size = batch_size
        self.image_crop_region = image_crop_region
        self.label_smoothing = label_smoothing
        self.input_processes = input_processes
        self.input_process_queue_size = input_process_queue_size
        self.read_input_threads = read_input_threads
        self.read_input_queue_size = read_input_queue_size
        self.valid_num_episodes = valid_num_episodes
        self.test_num_episodes = test_num_episodes
        self.positive_weight_target = positive_weight_target
        self.expected_positive_weight = expected_positive_weight
        self.clip_logits = clip_logits
        self.check_numerics = check_numerics
        self.false_positive_margin = false_positive_margin
        self.multiprocess = multiprocess
        self.verbose = verbose
        if kwargs:
            print("WARNING: Unknown hyperparameters: {}".format(kwargs))


class DataLoader:
    """Loads features from episode, and uses self.labeller to provide labels"""

    def __init__(self, labeller=None, hparams=TensorflowClassifierHparams()):
        self.labeller = labeller if labeller is not None else NullLabeller()
        self.hparams = hparams

    # TODO(): remove from DataLoader
    # numpy split array
    def split_episodes(self, episode_paths, n_train, n_valid, n_test, seed=None, use_all=True):
        """Split episodes between training, validation and test sets.

        seed: random seed (have split performed consistently every time)"""
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
            np.random.shuffle(episode_paths)
            np.random.set_state(random_state)
        else:
            np.random.shuffle(episode_paths)
        if use_all:
            multiplier = float(len(episode_paths)) / float(n_train + n_valid + n_test)
            n_train = int(math.floor(multiplier * n_train))
            n_valid = int(math.floor(multiplier * n_valid))
            n_test = int(math.floor(multiplier * n_test))

        assert n_train + n_valid + n_test <= len(episode_paths)
        return (episode_paths[:n_train], episode_paths[n_train:n_train + n_valid],
                episode_paths[n_train + n_test:n_train + n_test + n_test])

    def load_features_and_labels(self, episode_paths):
        features = {}
        labels = None
        for episode_path in episode_paths:
            try:
                suffix = "".join(pathlib.Path(episode_path).suffixes)
                if suffix == ".pkl.gz":
                    episode = frame.load_episode(episode_path)
                    episode_features, episode_labels = self.load_features_and_labels_episode(
                        episode)

                elif suffix == ".features":
                    with open(episode_path, 'rb') as f:
                        episode_features, episode_labels = pickle.load(f)
                else:
                    raise ValueError("Invalid suffix: {}".format(suffix))

                lengths = [len(value) for value in episode_features.values()]
                for length in lengths[1:]:
                    assert lengths[0] == length
                if suffix == ".pkl.gz":
                    episode_features['episode_path'] = np.array([episode_path] * lengths[0])

                for key, value in episode_features.items():
                    if key not in features:
                        features[key] = value
                    else:
                        features[key] = np.concatenate([features[key], value], axis=0)
                if labels is None:
                    labels = episode_labels
                else:
                    labels = np.concatenate([labels, episode_labels], axis=0)
            except Exception as e:
                print(episode_path)
                raise e
        return features, labels

    def load_features_episode(self, episode):
        features = {}
        observations = [frame.observation for frame in episode.frames if frame.has_action()]
        features['observation'] = np.concatenate(
            [np.expand_dims(observation, axis=0) for observation in observations], axis=0)
        images = [frame.image for frame in episode.frames if frame.has_action()]
        features['image'] = np.concatenate(
            [process_image(image, self.hparams) for image in images], axis=0)
        actions = [frame.get_proposed_action() for frame in episode.frames if frame.has_action()]
        features['action'] = np.expand_dims(np.array(actions), axis=1)
        features['index'] = np.array(
            [i for i, frame in enumerate(episode.frames) if frame.has_action()])
        return features

    def load_features_and_labels_episode(self, episode):
        features = self.load_features_episode(episode)
        return self.labeller.label(features, episode)

    def load_features_incident_records(self, incident_records):
        original_images = []
        actions = []
        for block_record_file in incident_records:
            with open(block_record_file, 'rb') as f:
                block_record = pickle.load(f)
                original_images.append(block_record.obs)
                actions.append(block_record.action)
        features = {}
        features['image'] = np.concatenate(
            [process_image(image, self.hparams) for image in original_images], axis=0)
        features['action'] = np.expand_dims(np.array(actions), axis=1)
        return features, original_images

    def predict_episodes(self, model, episode_paths, n=None, out_dir=None, prefix="model/"):
        if n is not None:
            episode_paths = np.random.choice(episode_paths, n, replace=False)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        for ep, episode_path in enumerate(episode_paths):
            episode = frame.load_episode(episode_path)
            features = self.load_features_episode(episode)
            prediction = model.predict_proba(features)
            for i in range(len(prediction)):
                episode.frames[i].info[prefix + "score"] = prediction[i]
                episode.frames[i].info[prefix + "label"] = model.apply_threshold(prediction[i])
            out_path = episode_path
            if out_dir is not None:
                out_path = os.path.join(out_dir, "{}.pkl.gz".format(ep))
            frame.save_episode(out_path, episode)


def process_image(image, hparams):
    desired_shape = hparams.image_shape
    if hparams.image_crop_region is not None:
        image = image[hparams.image_crop_region[0][0]:hparams.image_crop_region[0][1],
                      hparams.image_crop_region[1][0]:hparams.image_crop_region[1][1]]
    if not tuple(image.shape) == tuple(desired_shape):
        image = cv2.resize(image, (desired_shape[1], desired_shape[0]))
    assert tuple(image.shape) == tuple(desired_shape), "{}, {}".format(image.shape, desired_shape)
    return np.expand_dims(image.astype(np.float32) / 256.0, axis=0)


class NoisyLabellerWrapper:
    def __init__(self, labeller, false_positive_rate=0.0, false_negative_rate=0.0):
        self.labeller = labeller
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate

    def label(self, features, episode):
        features, labels = self.labeller.label(features, episode)
        return features, self.add_noise(labels, episode.path)

    def add_noise(self, labels, seed_str):
        if self.false_positive_rate > 0 or self.false_negative_rate > 0:
            r = np.random.RandomState()
            b = bytes(",".join(
                [seed_str, str(self.false_positive_rate),
                 str(self.false_negative_rate)]), 'ascii')
            r.seed(list(hashlib.md5(b).digest()))
            p = r.rand(len(labels))
            original_labels = np.copy(labels)
            labels[(~original_labels) & (p < self.false_positive_rate)] = True
            labels[(original_labels) & (p < self.false_negative_rate)] = False
        return labels

    ## currently unused, randomizes *noise_prob* of the labels
    def simple_add_noise(self, labels, seed_str):
        noise_prob = max(self.false_positive_rate, self.false_negative_rate)
        flip = lambda: np.random.choice([True, False])
        new_labels = []
        for label in labels:
            new_label = flip() if np.random.rand() > noise_prob else bool(label)
            new_labels.append(new_label)

        return np.array(new_labels)


class Labeller:
    def __init__(self):
        print('Constructing labeller')

    def label(self, features, episode):
        labels = np.array([False for frame in episode.frames if frame.has_action()])
        return features, labels


class AugmentedHumanBlockerLabeller(Labeller):
    def __init__(self, human_labeller, blocker_labeller):
        super(Labeller, self).__init__()
        self.blocker_labeller = blocker_labeller
        self.human_labeller = human_labeller

    def label(self, features, episode):
        features, labels = self.human_labeller.label(features, episode)
        if any(labels):
            return features, labels
        else:
            return self.blocker_labeller.label(features, episode)


class HumanLabeller:
    def __init__(self, frame_label='c', action_type=frame.Frame.PROPOSED):
        self.frame_label = frame_label
        self.action_type = action_type

    def _has_label(self, frame):
        return frame.get_label() == self.frame_label

    def label(self, features, episode):
        labels = np.array(
            [self._has_label(frame) for frame in episode.frames if frame.has_action()])
        actions = [
            frame.get_action(self.action_type) for frame in episode.frames if frame.has_action()
        ]
        features['action'] = np.expand_dims(np.array(actions), axis=1)
        return features, labels


class HumanOnlineBlockerLabeller(HumanLabeller):
    def __init__(self):
        super().__init__('b', frame.Frame.PROPOSED)


class HumanOfflineBlockerLabeller(HumanLabeller):
    def __init__(self):
        super().__init__('b', frame.Frame.REAL)


class HumanOfflineCatastropheLabeller(HumanLabeller):
    def __init__(self):
        super().__init__('c', frame.Frame.REAL)


class NullLabeller:
    def __init__(self):
        pass

    def label(self, features, episode):
        labels = np.array([False for frame in episode.frames if frame.has_action()])
        return features, labels


class BlockerLabeller:
    def __init__(self,
                 classifier=None,
                 info_entry=None,
                 block_radius=0,
                 catastrophe_array_fn=None,
                 only_catastrophe_starts=True):
        """
        Labeller for blocker, given a classifier

        Args:
            block_radius: if 0, looks only one step ahead for catastrophes
            only_catastrophe_starts: all frames that are already catastrophes
                are dropped, leaving only transistions from
                catastrophe->non-catastrophe as positive examples
        """
        self.classifier = classifier
        self.block_radius = block_radius
        self.info_entry = info_entry
        self.only_catastrophe_starts = only_catastrophe_starts
        self.catastrophe_array_fn = catastrophe_array_fn

    def label_and_build_mask(self, episode):
        if self.classifier is not None:
            is_catastrophe_array = np.array(
                self.classifier.is_catastrophe(
                    [frame.image for frame in episode.frames if frame.has_action()]))
        elif self.info_entry is not None:
            is_catastrophe_array = np.array(
                ([frame.info[self.info_entry] for frame in episode.frames if frame.has_action()]))
        elif self.catastrophe_array_fn is not None:
            is_catastrophe_array = self.catastrophe_array_fn(episode)
        else:
            raise ValueError("Don't know how to label catastrophes")

        labels = np.full(len(is_catastrophe_array), fill_value=False, dtype=np.bool)
        mask = np.full(len(is_catastrophe_array), fill_value=True, dtype=np.bool)

        for i in range(len(is_catastrophe_array)):
            if i + self.block_radius + 1 >= len(is_catastrophe_array):
                if self.only_catastrophe_starts:
                    mask[i] = False
                break
            if self.only_catastrophe_starts:
                if is_catastrophe_array[i]:
                    mask[i] = False
                    continue
            for j in range(self.block_radius + 1):
                if is_catastrophe_array[i + j + 1]:
                    labels[i] = True
                    break
        return labels, mask

    def label(self, features, episode):
        labels, mask = self.label_and_build_mask(episode)
        labels = labels[mask]
        for key in features.keys():
            features[key] = features[key][mask]
            assert (len(labels) == len(features[key])), "{} {}".format(
                len(labels), len(features[key]))
        return features, labels


def last_death_catastrophe_fn(episode):
    labels = [False] * range(len(episode.frames) - 1)
    for i in range(len(labels) - 1):
        current_lives = episode.frames[i].info.get("frame/lives")
        next_lives = episode.frames[i + 1].info.get("frame/lives")
        if (current_lives is not None and next_lives is not None and current_lives == 1 and
                next_lives == 0):
            labels[i + 1] = True
    return np.array(labels)


def death_catastrophe_fn(episode):
    labels = [False] * range(len(episode.frames) - 1)
    for i in range(len(labels) - 1):
        current_lives = episode.frames[i].info.get("frame/lives")
        next_lives = episode.frames[i + 1].info.get("frame/lives")
        if (current_lives is not None and next_lives is not None and current_lives > next_lives):
            labels[i + 1] = True
    return np.array(labels)


def get_unused_logdir(logdir):
    i = 1
    while os.path.isdir(logdir + str(i)):
        i += 1
    logdir = logdir + str(i)
    return logdir


def run_experiments(logdir,
                    data_loader,
                    datasets,
                    common_hparams,
                    hparams_list,
                    test_run=False,
                    predict_episodes=False,
                    **args):
    if not test_run:
        args2 = args.copy()
        args2["steps"] = 1
        temp_logdir = tempfile.mkdtemp()
        run_experiments(
            temp_logdir,
            data_loader, [d[:1] for d in datasets],
            common_hparams,
            hparams_list,
            test_run=True,
            **args2)
        shutil.rmtree(temp_logdir)
        if logdir is not None:
            if os.path.exists(logdir):
                shutil.rmtree(logdir)
            os.makedirs(os.path.realpath(logdir))
            subprocess.call("pkill tensorboard", shell=True)
            subprocess.call(
                "tensorboard --logdir {logdir} --reload_interval 1 --port 12345 &".format(
                    logdir=logdir),
                shell=True)
    for i, hparams in enumerate(hparams_list):
        hparams.update(common_hparams)
        g = tf.Graph()
        with g.as_default():
            classifier = TensorflowClassifier(data_loader, TensorflowClassifierHparams(**hparams))
            sess = tf.Session(graph=g)
            with sess.as_default():
                rundir = None
                if logdir is not None:
                    rundir = os.path.join(logdir, str(i))
                    os.makedirs(rundir, exist_ok=False)
                classifier.fit(*datasets, logdir=rundir, **args)
                # if rundir is not None:
                #     if predict_episodes:
                #         data_loader.predict_episodes(classifier,
                #                                      datasets[0] + datasets[1] + datasets[2])
            sess.close()
    subprocess.call("pkill tensorboard", shell=True)


if __name__ == "__main__":
    tp = unittest.main(exit=False)
    if not tp.result.wasSuccessful():
        sys.exit(False)

    import pong_catastrophe
    from pong_catastrophe import PongClassifierLabeller, PongBlockerLabeller

    episode_paths = frame.episode_paths("logs/PongDeterministic-v3-Episodes")
    hparams = TensorflowClassifierHparams(
        use_image=False,
        use_observation=True,
        batch_size=400,
        image_shape=[210, 160, 3],
        multiprocess=False,
        convolution2d_stack_args=[(16, [3, 3], [2, 2])] * 6)
    #   image_crop_region=((34,34+160),(0,160)))
    data_loader = DataLoader(PongBlockerLabeller())
    datasets = data_loader.split_episodes(episode_paths, 1, 1, 1, use_all=False)
    g = tf.Graph()
    with g.as_default():
        classifier = TensorflowClassifier(data_loader, hparams)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(graph=g, config=config)
        with sess.as_default():
            classifier.fit(*datasets, steps=100, logdir='/tmp/foo/blockerfast', log_every=10)
            # classifier.save(checkpoint_name="/tmp/foo/blocker/0.ckpt")
        sess.close()
