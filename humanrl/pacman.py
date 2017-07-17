import gym
import numpy as np

import cv2

LIFE_ROW = 174
LIFE_WIDTH = 6

def count_lives(frame):
    count = 0
    for v in frame[LIFE_ROW][:80]:
        if v > 0:
            count += 1
    return count // LIFE_WIDTH

def downsample(frame):
    # input is 210 x 160
    frame = frame[1:172]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class Pacman(gym.Wrapper):
    def __init__(self, death_penalty=True, deterministic=True, v=3, **kwargs):
        env_id = "MsPacman"
        if deterministic:
            env_id += "Deterministic"
        env_id += "-v%d" % v

        env = gym.make(env_id)
        super(Pacman, self).__init__(env)
        self.observation_space = gym.spaces.Box(0.0, 1.0, [42, 42, 1])
        self.death_penalty = death_penalty

    def _reset(self):
        self.lives = 5
        obs = self.env.reset()
        obs = obs.mean(2)
        return downsample(obs)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward /= 100

        obs = obs.mean(2) # greyscale

        if self.death_penalty:
            n = count_lives(obs)
            lost = self.lives - n
            self.lives = n

            if done:
                lost += 1

            if lost > 0:
                reward -= lost

        #info["lives"] = n
        obs = downsample(obs)

        return obs, reward, done, info

if __name__ == "__main__":
    import time
    env = gym.make("MsPacman-v3")
    env = PacmanWrapper(env)
    observation = env.reset()

    for _ in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print(
        #    paddle_top(observation),
        #    paddle_bottom(observation), action, reward, "CATASTROPHE"
        #    if env.is_catastrophe(observation) else "", "PRUNED"
        #    if env.last_action_pruned else "")
        #time.sleep(0.1)

        print(reward)
        #print(observation)

        # print_observation(observation)
