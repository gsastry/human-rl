#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import time

import gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "humanrl"))

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

env = gym.make('Pong-v3' if len(sys.argv) < 2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
ROLLOUT_TIME = 1000
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
# can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xff0d: human_wants_restart = True
    if key == 32: human_sets_pause = not human_sets_pause
    if key == ord('q'): exit()
    print(key)
    if key == 37:
        human_agent_action = 3
    a = int(key - ord('0'))
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    a = int(key - ord('0'))
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    for t in range(ROLLOUT_TIME):
        if not skip:
            print("taking action {}".format(human_agent_action))
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        env.render()
        if done: break
        if human_wants_restart: break
        time.sleep(0.1)
        while human_sets_pause:
            env.render()

            time.sleep(0.1)


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    rollout(env)
