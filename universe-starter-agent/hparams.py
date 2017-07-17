"""Single place to store definition of new hyperparameters for agent training"""
import ast

import go_vncdriver

from humanrl.pong_catastrophe import (DEFAULT_BLOCK_CLEARANCE, DEFAULT_CLEARANCE, DEFAULT_LOCATION)
from model import policies

# Definitions for agent training hyperparameters. Key is name of hyperparameter, which
# becomes an optional keyword argument (prefixed by -- on the command line). dictionary
# is a dictionary of keywords passed to parser.add_argument (from argparse module)
HPARAM_DEFINITIONS = {
    'clearance':
    dict(
        default=DEFAULT_CLEARANCE,
        type=int,
        help="Pong, if distance in pixels from the top or bottom of the "
        "screen is less than clearance, state is a catastrophe"),
    'block_clearance':
    dict(
        default=DEFAULT_BLOCK_CLEARANCE,
        type=int,
        help="Pong, heuristic blocker blocks when the distance from bottom of "
        "screen is less than clearance + block_clearance"),
    'location':
    dict(
        default=DEFAULT_LOCATION,
        choices=["top", "bottom"],
        help="Pong, whether catastrophe is defined as being close to the top or "
        " bottom of the screen"),
    'learning_rate':
    dict(default=0.0001, type=float, help="A3C"),
    "max_episodes":
    dict(
        default=50,
        type=ast.literal_eval,
        help="A3C: Maximum number of episodes (per worker) to record. If none, record all episodes"
    ),
    "classifier_file":
    dict(
        default="",
        help=
        "tensorflow checkpoint storing classifier (or list of checkpoints if an ensemble is used)",
        nargs="+"),
    "blocker_file":
    dict(
        default="",
        help="tensorflow checkpoint storing blocker (or list of checkpoints if an ensemble is used)",
        nargs="+"),
    "classifier_threshold":
    dict(default=None, type=ast.literal_eval, help="overrides classifier threshold"),
    "blocker_threshold":
    dict(default=None, type=ast.literal_eval, help="overrides blocker threshold"),
    "blocker_threshold_relative":
    dict(default=None, type=ast.literal_eval),
    "allowed_actions_source":
    dict(
        default="blocker",
        choices=["blocker", "heuristic"],
        help="Source of allowed actions when blocking"),
    "blocking_mode":
    dict(
        default="none",
        choices=["none", "action_pruning", "action_replacement", "observe", "penalty_only"],
        help="Method of blocking. action_replacement replaces action with a "
        "randomly selected safe action, action_pruning bounces back to the "
        "agent in the same state until the agent submits a safe action, "
        "observe doesn't apply blocking, but logs the number of times it "
        "would have happened"),
    "online_blocking_mode":
    dict(
        default="none",
        choices=["none", "action_pruning", "action_replacement", "observe", "penalty_only"],
        help="Method of blocking. action_replacement replaces action with a "
        "randomly selected safe action, action_pruning bounces back to the "
        "agent in the same state until the agent submits a safe action, "
        "observe doesn't apply blocking, but logs the number of times it "
        "would have happened"),
    "reward_scale":
    dict(
        default=1.0,
        type=float,
        help="All rewards (including catastrophe reward) are DIVIDED by this amount"),
    "catastrophe_type":
    dict(
        default="",
        type=str,
        help="Catastrophe type for pong. "
        "By default this is null, which is equivalent to turning "
        "off the catastrophe wrapper"),
    "catastrophe_reward":
    dict(default=0.0, type=float, help="reward given when catastrophe occurs"),
    "log_block_catastrophes":
    dict(
        default=True,
        type=ast.literal_eval,
        help="Whether to write pickle files containing records of block and catastrophe events"),
    "log_reward":
    dict(
        default=False,
        type=ast.literal_eval,
        help="Whether to log reward at every step (makes tensorflow event files large)"),
    "no_jump":
    dict(default=False, type=ast.literal_eval, help="No jump"),
    "online":
    dict(default=False, type=ast.literal_eval),
    "lose_first_two_lives":
    dict(
        default=False,
        type=ast.literal_eval,
        help="On roadrunner, if set to true run the environment with no-ops until "
        "first two lives are lost, then hand over control to the agent"),
    "max_level_catastrophe_wrapper_active":
    dict(
        default=None,
        type=ast.literal_eval,
        help="Maximum level that blocking/catastrophe detection are active on (no effect if None)"),
    "max_level":
    dict(
        default=None,
        type=ast.literal_eval,
        help=
        "In roadrunner, if level goes beyond this then the episode terminates (no effect if None)"),
    "max_episode_blocks":
    dict(default=None, type=ast.literal_eval),
    "unblocked_bonus":
    dict(default=False, type=ast.literal_eval),
    "squash_rewards":
    dict(default=False, type=ast.literal_eval, help="If true, squash all rewards to +/-1"),
    "render":
    dict(default=True, type=ast.literal_eval, help="Try to render environment"),
    "extra_wrapper":
    dict(
        default="",
        type=str,
        help="If set, add an extra wrapper to the environment (hack for space invaders)"),

    # exploration
    "explore":
    dict(type=int, default=0, help="give exploration bonus"),
    "explore_scale":
    dict(type=float, default=1., help="exploration bonus scale factor"),
    "explore_buffer":
    dict(type=float, default=1e4, help="total state visitation counts/memory"),
    "decay":
    dict(type=int, default=0, help="decay visitation counts"),
    "gamma":
    dict(type=float, default=0.99, help="reward discount factor"),
    "lstm_size":
    dict(type=int, default=256, help="lstm state size"),
    "entropy_scale":
    dict(type=float, default=0.01, help="scale of the entropy bonus"),
    "clip_norm":
    dict(type=float, default=40.0, help="global variable norm at which to clip gradients"),
    "local_steps":
    dict(type=int, default=20, help="number of steps between policy gradient updates"),
    "policy":
    dict(type=str, default="lstm", choices=policies.keys(), help="policy type"),
    "death_penalty":
    dict(type=float, default=0, help="Impose penalty for dying. Relevant to Pacman, Montezuma."),
    "deterministic":
    dict(type=int, default=1, help="Use deterministic environment."),
    "use_categorical_max":
    dict(
        type=ast.literal_eval,
        default=False,
        help="Whether to always take the max prob action from the policy"),
}


def add_hparams(parser):
    """Adds hyperparameters to an argparse.ArgumentParser object"""
    for k, v in HPARAM_DEFINITIONS.items():
        parser.add_argument("--" + k, **v)
    return parser


def get_hparams(args, ignore_default=False):
    """Returns dictionary of hyperparameters from parsed arguments"""
    # Only return the arguments that are added by hparams
    if ignore_default:
        hparams = {
            k: args.__dict__[k]
            for k, v_dict in HPARAM_DEFINITIONS.items() if args.__dict__[k] is not v_dict['default']
        }
    else:
        hparams = {k: args.__dict__[k] for k in HPARAM_DEFINITIONS.keys()}
    return hparams
