import argparse
import os
import sys

import numpy as np

from humanrl.space_invaders import (SpaceInvadersStartsWrapper,
                                    SpaceInvadersStartsWrapper2)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))
import frame  # isort:skip

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument(
    '-f',
    '--frames-dir',
    type=str,
    default="/tmp/pong/frames",
    help="Directory to read and write frames to")
parser.add_argument('--env-id', type=str, default="Pong", help="ID of environment to run")
parser.add_argument(
    '-n', '--num-episodes', type=int, default=100, help="Number of episodes to generate")
parser.add_argument('--extra_wrapper', type=str, default="", help="Special wrapper to add")

if __name__ == "__main__":
    import go_vncdriver
    args = parser.parse_args()
    from universe_starter_agent import envs
    env = envs.create_env(args.env_id, extra_wrapper=args.extra_wrapper)

    env = frame.FrameSaverWrapper(env, args.frames_dir, max_episodes=None)
    for _ in range(args.num_episodes):
        observation = env.reset()
        for i in range(10000):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
    episode_paths = frame.episode_paths(args.frames_dir)
    episodes = [frame.load_episode(episode_path) for episode_path in episode_paths]
    print("Number of episodes: {}".format(len(episodes)))
