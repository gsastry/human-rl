import logging
import os
import pickle
from collections import namedtuple

import gym
import numpy as np

from catastrophe_wrapper import *
from catastrophe_wrapper import CatastropheWrapper
from classifier_tf import (SavedCatastropheBlockerTensorflow,
                           SavedCatastropheClassifierTensorflow)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TOLERANCE = 0.01
PADDLE_COLUMN = {"right": 143, "left": 16}
PADDLE_COLOR = {"right": np.array([92, 186, 92]), "left": np.array([213, 130, 74])}
PLAY_AREA = [34, 34 + 160]
DEFAULT_CLEARANCE = 16
DEFAULT_BLOCK_CLEARANCE = 16
DEFAULT_LOCATION = "bottom"


def paddle_bottom(observation, paddle="right"):
    column = observation[:, PADDLE_COLUMN[paddle], :] - PADDLE_COLOR[paddle]
    found = (np.sum(np.abs(column), axis=1) < TOLERANCE).astype(np.int)
    r = np.argmax(np.flipud(found))
    r = (len(found) - r - 1)
    if not found[r]:
        return None
    else:
        return r


def paddle_top(observation, paddle="right"):
    column = observation[:, PADDLE_COLUMN[paddle], :] - PADDLE_COLOR[paddle]
    found = (np.sum(np.abs(column), axis=1) < TOLERANCE).astype(np.int)
    r = np.argmax(found)
    if not found[r]:
        return None
    else:
        return r


# def ball_center(observation):
#     w = np.where(np.abs(observation[:,6:36] - 0.30457518) > TOLERANCE)[:2]
#     if len(w[0]) == 0 or len(w[0]) > 4:
#         return None
#     w = np.mean(w, axis=1)
#     return w[0], w[1] + 6
#
# def ball_on_left(observation):
#     w = np.where(np.abs(observation[:,6:21] - 0.30457518) > TOLERANCE)[:2]
#     return(len(w[0]) > 0)


def print_observation(observation):
    L = []
    for row in observation:
        for value in row:
            found = False
            for i, v in enumerate(L):
                if (v == value).all():
                    found = True
                    print(i, end="")
                    break
            if not found:
                print(len(L), end="")
                L.append(value)
        print()
    for i, v in enumerate(L):
        print(i, v)


def is_catastrophe(obs, location=DEFAULT_LOCATION, clearance=DEFAULT_CLEARANCE):    
    
    if location == "bottom":
        y = paddle_bottom(obs)
        if y is None:
            return False
        return y > PLAY_AREA[1] - clearance
    elif location == "top":
        y = paddle_top(obs)
        if y is None:
            return False
        return y < PLAY_AREA[0] + clearance
    else:
        raise ValueError("Unrecognized location: {}".format(location))

##catastrophe zone on either side
# def is_catastrophe(obs, location=DEFAULT_LOCATION, clearance=DEFAULT_CLEARANCE):
#     clearance = 5
#     y = paddle_bottom(obs)
#     y2 = paddle_top(obs)
#     if y is None:
#         return False
#     return (y > PLAY_AREA[1] - clearance) or (y < PLAY_AREA[0] + clearance)
    


def should_block(obs,
                 action,
                 location=DEFAULT_LOCATION,
                 clearance=DEFAULT_CLEARANCE,
                 block_clearance=DEFAULT_BLOCK_CLEARANCE):
    if obs is None:
        return False
    if is_catastrophe(obs, location, clearance):
        return False
    if location == "top":
        # DOWN
        if is_catastrophe(obs, location, clearance + block_clearance) and action != 5:
            return True
    elif location == "bottom":
        # UP
        if is_catastrophe(obs, location, clearance + block_clearance) and action != 2:
            return True
    return False


def allowed_actions_heuristic(obs, location):
    if location == "top":
        return [5]
    elif location == "bottom":
        return [2]


class CatastropheClassifierHeuristic(object):
    def __init__(self, location=DEFAULT_LOCATION, clearance=DEFAULT_CLEARANCE, **_):        
        self.location = location
        self.clearance = clearance

    def is_catastrophe(self, obs):       
        return is_catastrophe(obs, self.location, self.clearance)

    def is_catastrophe_with_score(self, obs):
        return is_catastrophe(obs, self.location, self.clearance), 0


# class CatastropheClassifierHeuristic2(object):
#     def __init__(self):
#         pass
#
#     def is_catastrophe(self, obs):
# #         if True:
#         if ball_on_left(obs):
#             ltop, lbottom = paddle_top(obs, "left"), paddle_bottom(obs, "left")
#             rtop, rbottom = paddle_top(obs, "right"), paddle_bottom(obs, "right")
#             print([ltop, lbottom, rtop, rbottom])
#             if None not in [ltop, lbottom, rtop, rbottom]:
#
#                 if rbottom >= ltop and rbottom <= lbottom:
#                     return True
#                 if rtop >= ltop and rtop <= lbottom:
#                     return True
#                 if rtop < ltop and rbottom > lbottom:
#                     return True
#         return False


class CatastropheBlockerHeuristic(object):
    def __init__(self,
                 location=DEFAULT_LOCATION,
                 clearance=DEFAULT_CLEARANCE,
                 block_clearance=DEFAULT_BLOCK_CLEARANCE,
                 **_):
        self.location = location
        self.clearance = clearance
        self.block_clearance = block_clearance

    def should_block(self, obs, action):
        return should_block(obs, action, self.location, self.clearance, self.block_clearance)

    def should_block_with_score(self, obs, action):
        return should_block(obs, action, self.location, self.clearance, self.block_clearance), 0

    def allowed_actions(self, obs):
        return allowed_actions_heuristic(obs, self.location)


class PongClassifierLabeller(object):
    def __init__(self):
        pass

    def label(self, features, episode):
        images = (frame.image for frame in episode.frames if frame.action is not None)
        labels = np.array([is_catastrophe(image, location="bottom") for image in images])
        return features, labels


class PongBlockerClearanceHeuristicLabeller(object):
    def __init__(self,
                 location=DEFAULT_LOCATION,
                 clearance=DEFAULT_CLEARANCE,
                 block_clearance=DEFAULT_BLOCK_CLEARANCE,
                 **_):
        self.location = location
        self.clearance = clearance
        self.block_clearance = block_clearance
        self.blocker = CatastropheBlockerHeuristic(location, clearance, block_clearance)

    def __block_with_clearance(self, obs, action, location, clearance, block_clearance):
        if is_catastrophe(obs, location, clearance + block_clearance) and action != 2:   # 'up' action
            return True
        else:
            return False

    def label(self, features, episode):
        labels = np.array(
            [self.__block_with_clearance(frame.image,
                                         frame.action,
                                         self.location,
                                         self.clearance,
                                         self.block_clearance)
             for frame in episode.frames if frame.action is not None])
        return features, labels

class PongBlockerLabeller(object):
    def __init__(self, block_radius=0):
        self.block_radius = block_radius

    def label_and_build_mask(self, episode):
        is_catastrophe_array = np.array(
            [is_catastrophe(frame.image) for frame in episode.frames if frame.action is not None])
        # should_block_array = np.array([should_block(frame.image, frame.action) for frame in episode.frames])

        labels = np.full(len(episode.frames), fill_value=False, dtype=np.bool)
        mask = np.full(len(episode.frames), fill_value=True, dtype=np.bool)

        for i in range(len(episode.frames)):
            if i + self.block_radius + 1 >= len(episode.frames):
                mask[i] = False
                continue
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
        for key, value in features.items():
            features[key] = features[key][mask]
            assert (len(labels) == len(features[key])), "{} {}".format(
                len(labels), len(features[key]))
        return features, labels
        # return features, labels
