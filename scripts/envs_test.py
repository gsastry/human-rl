import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

import go_vncdriver  # isort:skip
from universe_starter_agent import envs  # isort:skip

if __name__ == "__main__":
    # import time
    # env = envs.create_env(
    #     "RoadRunner",
    #     catastrophe_type="1",
    #     classifier_file="models/roadrunner/c1/classifier/final.ckpt")
    #
    # observation = env.reset()
    #
    # for _ in range(1):
    #     action = env.action_space.sample()
    #     observation, reward, done, info = env.step(action)
    #     print(info)
    #     print(reward)
    #     if done:
    #         break

    env = envs.create_env(
        "Pong",
        location="bottom",
        catastrophe_type="1",
        classifier_file="models/pong/c1/classifier/final.ckpt",
        blocker_file="models/pong/c1/blocker/final.ckpt",
        blocking_mode="action_replacement")

    observation = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # if env.last_obs is not None:
        #     print(
        #         pong_catastrophe.paddle_top(env.last_obs),
        #         pong_catastrophe.paddle_bottom(env.last_obs), action, reward, "CATASTROPHE"
        #         if env.is_catastrophe(env.last_obs) else "", "BLOCKED"
        #         if env.last_action_blocked else "")
        print(info)
        if done:
            break
