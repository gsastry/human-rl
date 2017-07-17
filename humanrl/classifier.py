import pickle

import numpy as np


class CatastropheClassifier:
    def is_catastrophe(self, obs):
        return False


class CatastropheClassifierSKLearn:
    def __init__(self, classifier_file):
        with open(classifier_file, "rb") as f:
            self.classifier = pickle.load(f)

    def is_catastrophe(self, obs):
        X = np.expand_dims(obs, axis=0)
        X = np.reshape(X, [X.shape[0], -1])
        return self.classifier.predict(X)[0]


class CatastropheBlocker:
    def should_block(self, obs, action):
        pass

    def should_block_with_score(self, obs, action):
        pass

    def allowed_actions(self, obs):
        pass

    def allowed_actions_with_scores(self, obs):
        pass
