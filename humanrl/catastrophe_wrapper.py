import logging
import pickle
from collections import namedtuple

import gym
import numpy as np

import log_coordinator
from utils import SAFE_ACTION_MAPPINGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CatastropheRecord = namedtuple('CatastropheRecord', [
    'obs', 'last_obs', 'action', 'last_action_blocked', 'obs_catastrophe', 'last_obs_catastrophe'
])
CatastropheRecord.__new__.__defaults__ = (None, ) * len(CatastropheRecord._fields)

BlockRecord = namedtuple('BlockRecord', [
    'obs', 'action', 'last_action_blocked', 'next_obs', 'next_obs_catastrophe', 'action_scores'
])
BlockRecord.__new__.__defaults__ = (None, ) * len(BlockRecord._fields)


class CatastropheWrapper(gym.Wrapper):
    """
*catastrophe_reward* is also reward agents gets when blocked (for all modes)
*blocking_mode*: 'observe' means no blocking (but recording when blocking would have occurred).
'action_pruning' blocks and then allows the agent to try again from the previous observation.
*blocker*: blocker object built in envs.py using *SavedCatastropheBlockerTensorflow* from 'blocker_file'
argument to train.py.
*allowed_actions_heuristic*: must be given if *allowed_actions_source*='heuristic'

When testing, try running with:
Unmodified environment
Classifier
Classifier + baseline
blocker (action_pruning)
blocker (action_replacement)
blocker + classifier
"""

    class Counters:
        def __init__(self):
            self.episode_catastrophe_count = 0
            self.episode_reward_with_catastrophes = 0
            self.episode_reward_without_catastrophes = 0
            self.episode_block_count = 0
            self.episode_catastrophe_start_count = 0
            self.block_number = 0
            self.classifier_catastrophe_count = 0
            self.false_positive_count = 0
            self.false_negative_count = 0
            self.cumulative_block_count = 0

        def as_dict(self, include_classifier_baseline_counters=True):
            info = {}
            info["global/episode_reward_with_catastrophes"] = self.episode_reward_with_catastrophes
            info[
                "global/episode_reward_without_catastrophes"] = self.episode_reward_without_catastrophes
            info["global/episode_catastrophe_count"] = self.episode_catastrophe_count
            info["global/episode_catastrophe_start_count"] = self.episode_catastrophe_start_count
            info["global/episode_block_count"] = self.episode_block_count

            if include_classifier_baseline_counters:
                info["global/classifier_catastrophe_count"] = self.classifier_catastrophe_count
                info["global/false_positive_count"] = self.false_positive_count
                info["global/false_negative_count"] = self.false_negative_count

            return info

    def __init__(self,
                 env,
                 catastrophe_reward=-1,
                 blocking_mode="none",
                 catastrophe_dir="",
                 block_dir="",
                 log_block_catastrophes=True,
                 classifier=None,
                 blocker=None,
                 allowed_actions_source="blocker",
                 allowed_actions_heuristic=None,
                 classifier_baseline=None,
                 max_catastrophe_records=50,
                 max_block_records=50,
                 max_level_catastrophe_wrapper_active=None,
                 max_episode_blocks=None,
                 unblocked_bonus=False,
                 squash_rewards=False,
                 safe_action_mapping=None,
                 **_):
        super(CatastropheWrapper, self).__init__(env)
        self.catastrophe_reward = catastrophe_reward
        self.blocking_mode = blocking_mode
        self.last_obs = None
        self.last_info = None
        self.last_obs_catastrophe = False
        self.last_action_blocked = False
        self.cumulative_block_count = 0
        self.catastrophe_coordinator = None
        self.block_coordinator = None
        self.max_level_catastrophe_wrapper_active = max_level_catastrophe_wrapper_active
        self.active = True
        self.max_episode_blocks = max_episode_blocks
        self.unblocked_bonus = unblocked_bonus
        self.squash_rewards = squash_rewards
        self.safe_action_mapping = safe_action_mapping
        if log_block_catastrophes:
            if catastrophe_dir:
                self.catastrophe_coordinator = log_coordinator.LogCoordinator(
                    max_length=max_catastrophe_records,
                    logdir=catastrophe_dir,
                    filename_str="{i}.pkl")
            if block_dir:
                self.block_coordinator = log_coordinator.LogCoordinator(
                    max_length=max_block_records, logdir=block_dir, filename_str="{i}.pkl")
        self.allowed_actions_source = allowed_actions_source
        self.allowed_actions_heuristic = allowed_actions_heuristic
        assert not (self.allowed_actions_source == "heuristic" and
                    self.allowed_actions_heuristic is None), self.allowed_actions_heuristic
        self.classifier_baseline = classifier_baseline
        self.classifier = classifier
        self.blocker = blocker
        self.counters = CatastropheWrapper.Counters()

    def _reset(self):
        obs = self.env.reset()
        self.active = True
        self.last_obs_catastrophe = False
        self.last_action_blocked = False
        self.counters = CatastropheWrapper.Counters()
        return obs

    def write_catastrophe_record(self, *args, **kwargs):
        if self.catastrophe_coordinator is not None:
            if self.catastrophe_coordinator.should_log():
                record = CatastropheRecord(*args, **kwargs)
                filename = self.catastrophe_coordinator.get_filename()
                with open(filename, "wb") as f:
                    pickler = pickle.Pickler(f)
                    pickler.dump(record)
            self.catastrophe_coordinator.step()

    def write_block_record(self, *args, **kwargs):
        if self.block_coordinator is not None:
            if self.block_coordinator.should_log():
                record = BlockRecord(*args, **kwargs)
                filename = self.block_coordinator.get_filename()
                with open(filename, "wb") as f:
                    pickler = pickle.Pickler(f)
                    pickler.dump(record)
            self.block_coordinator.step()

    def handle_blocking(self, proposed_action):
        real_action = proposed_action
        extra_info = {}
        extra_reward = 0
        if self.blocking_mode != "none":
            should_block, should_block_score = self.should_block(self.last_obs, proposed_action)
            if should_block:
                self.last_action_blocked = True
                self.counters.episode_block_count += 1
                self.cumulative_block_count += 1
                if not self.last_obs_catastrophe:
                    allowed_actions, block_scores = self.allowed_actions_with_scores(self.last_obs)
                    self.write_block_record(self.last_obs, proposed_action,
                                            self.last_action_blocked, None, None, block_scores)
                extra_info["frame/action/should_block"] = should_block
                extra_info["frame/action/should_block_score"] = should_block_score
                extra_info["frame/action/proposed_action"] = proposed_action

                if self.blocking_mode == "action_pruning":
                    extra_reward = self.catastrophe_reward
                    if proposed_action not in allowed_actions:
                        real_action = None
                elif self.blocking_mode == "action_replacement":
                    real_action = self.safe_policy(self.last_obs, proposed_action)
                    extra_reward = self.catastrophe_reward
                elif self.blocking_mode == "penalty_only":
                    extra_reward = self.catastrophe_reward
                elif self.blocking_mode == "observe":
                    pass
                else:
                    raise ValueError("Invalid blocking_mode: {}", self.blocking_mode)
                extra_info["frame/action/real_action"] = real_action
            else:
                if self.unblocked_bonus and self.cumulative_block_count > 0:
                    extra_reward = (-self.catastrophe_reward * (self.cumulative_block_count - 1))
                self.cumulative_block_count = 0
                self.last_action_blocked = False

        return real_action, extra_reward, extra_info

    def handle_catastrophe(self, obs, action):
        extra_info = {}
        extra_reward = 0
        is_catastrophe = False
        if self.classifier is not None:
            is_catastrophe, is_catastrophe_score = self.classifier.is_catastrophe_with_score(obs)
            extra_info["frame/is_catastrophe"] = is_catastrophe
            extra_info["frame/is_catastrophe_score"] = is_catastrophe_score

            if is_catastrophe:
                extra_reward = self.catastrophe_reward
                self.counters.episode_catastrophe_count += 1
                if not self.last_obs_catastrophe:
                    self.counters.episode_catastrophe_start_count += 1
                    self.write_catastrophe_record(obs, self.last_obs, action,
                                                  self.last_action_blocked, is_catastrophe,
                                                  self.last_obs_catastrophe)

            if self.classifier_baseline is not None:
                is_catastrophe_baseline = self.classifier_baseline.is_catastrophe(obs)
                is_catastrophe_classifier = self.classifier.is_catastrophe(obs)
                if is_catastrophe_classifier:
                    self.counters.classifier_catastrophe_count += 1
                if is_catastrophe_baseline and not is_catastrophe_classifier:
                    self.counters.false_negative_count += 1
                if not is_catastrophe_baseline and is_catastrophe_classifier:
                    self.counters.false_positive_count += 1
        self.last_obs_catastrophe = is_catastrophe
        return extra_reward, extra_info

    def check_if_still_active(self, info):
        if (self.max_level_catastrophe_wrapper_active is not None and "frame/level" in info and
                info["frame/level"] > self.max_level_catastrophe_wrapper_active):
            self.active = False

    def _step(self, proposed_action):
        if self.active:
            real_action, extra_reward_blocking, extra_info = self.handle_blocking(proposed_action)

            if real_action is None:
                obs, reward, done, info = self.last_obs, 0, False, self.last_info.copy()
            else:
                obs, reward, done, info = self.env.step(real_action)
            self.last_info = info

            # Hack for space invaders logging (William)
            if info.get("frame/should_have_blocked", False):
                self.write_catastrophe_record(obs, self.last_obs, real_action,
                                              self.last_action_blocked, True, None)
                self.counters.episode_catastrophe_count += 1

            if self.squash_rewards:
                reward = float(np.sign(reward))

            self.counters.episode_reward_without_catastrophes += reward

            info.update(extra_info)
            reward += extra_reward_blocking

            self.check_if_still_active(info)

            extra_reward_catastrophe, extra_info = self.handle_catastrophe(obs, real_action)
            info.update(extra_info)
            reward += extra_reward_catastrophe

            if self.squash_rewards:
                if (extra_reward_blocking < 0 or extra_reward_catastrophe < 0):
                    reward = -1.0
                else:
                    reward = float(np.sign(reward))

            # If we exceed the maximum number of blocks per episode then terminate the episode
            if (self.max_episode_blocks is not None and
                    self.counters.episode_block_count > self.max_episode_blocks):
                done = True

            self.last_obs = obs
        else:
            obs, reward, done, info = self.env.step(proposed_action)

        self.counters.episode_reward_with_catastrophes += reward

        if done:
            include_classifier_baseline_counters = self.counters.as_dict(
                self.classifier is not None and self.classifier_baseline is not None)
            info.update(include_classifier_baseline_counters)

        return obs, reward, done, info

    def should_block(self, obs, proposed_action):
        if obs is None:
            return False, 0
        if self.safe_action_mapping is not None:
            # If action is the action we would take if blocking, then we just take that action
            if proposed_action == SAFE_ACTION_MAPPINGS[self.safe_action_mapping][proposed_action]:
                return False, 0
        return self.blocker.should_block_with_score(obs, proposed_action)

    def allowed_actions_with_scores(self, obs):
        if self.allowed_actions_source == "heuristic":
            return self.allowed_actions_heuristic(obs), None
        elif self.allowed_actions_source == "blocker":
            return self.blocker.allowed_actions_with_scores(obs)
        else:
            raise ValueError(
                "Invalid allowed_actions_source: {}".format(self.allowed_actions_source))

    def allowed_actions(self, obs):
        return self.allowed_actions_with_scores(obs)[0]

    def safe_policy(self, obs, proposed_action=None):
        if self.safe_action_mapping is not None:
            return SAFE_ACTION_MAPPINGS[self.safe_action_mapping][proposed_action]
        return np.random.choice(self.allowed_actions(obs))
