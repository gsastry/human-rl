import gym
import numpy as np

from humanrl import frame

BARRIER_COLOR = [181, 83, 40]
BARRIER_ROWS = (157, 175)
INITIAL_BARRIER_PIXELS = 336
SHIP_COLOR = [50, 132, 50]
SHIP_ROWS = (185, 195)


def barrier_damage(image):
    is_barrier = np.sum(np.abs(image - BARRIER_COLOR), axis=2) == 0
    return INITIAL_BARRIER_PIXELS - np.sum(is_barrier)


def barriers_gone(image):
    is_barrier = np.sum(np.abs(image - BARRIER_COLOR), axis=2) == 0
    return np.sum(is_barrier) == 0


def ship_location(image):
    is_ship = np.sum(np.abs(image[185, :, :] - SHIP_COLOR), axis=1) == 0
    w = np.argwhere(is_ship)
    return w[0][0] if len(w) == 1 else None


def is_below_barrier(image, x1, x2):
    is_barrier = np.sum(
        np.abs(image[BARRIER_ROWS[0]:BARRIER_ROWS[1], x1:(x2 + 1), :] - BARRIER_COLOR), axis=2) == 0
    return np.sum(is_barrier) > 0


def is_ship_below_barrier(image, margin=0):
    x = ship_location(image)
    if x is None:
        return False
    return is_below_barrier(image, x - margin, x + margin)


def should_block(image, action, margin=0):
    return action in (1, 4, 5) and is_ship_below_barrier(image, margin)


class SpaceInvadersStartsWrapper(gym.Wrapper):
    def _reset(self):
        obs = self.env.reset()
        start_mode = np.random.choice(4)
        if start_mode == 1:
            for i in range(48):
                obs, _, _, _ = self.env.step(2)
        elif start_mode == 2:
            for i in range(69):
                obs, _, _, _ = self.env.step(2)
        elif start_mode == 3:
            for i in range(94):
                obs, _, _, _ = self.env.step(0 if i < 45 else 2)
        if start_mode > 0:
            done = False
            if np.random.random() > 0.5:
                while not barriers_gone(obs) and not done:
                    obs, _reward, done, _info = self.env.step(0)
        return obs


class SpaceInvadersStartsWrapper2(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.start_mode = np.random.randint(3)
        self.max_steps_to_wait = 800

    def _reset(self):
        obs = self.env.reset()
        if self.start_mode == 0:
            for i in range(48):
                obs, _, _, _ = self.env.step(2)
        elif self.start_mode == 1:
            for i in range(69):
                obs, _, _, _ = self.env.step(2)
        elif self.start_mode == 2:
            for i in range(94):
                obs, _, _, _ = self.env.step(0 if i < 45 else 2)
        self.start_mode = (self.start_mode + 1) % 3
        to_wait = np.random.randint(0, self.max_steps_to_wait)
        for i in range(to_wait):
            obs, _reward, done, _info = self.env.step(0)
            if done:
                break
        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = info.get("ale.lives", None)
        if lives is not None and lives < 3:
            done = True
        return obs, reward, done, info


class SpaceInvadersMonitoringWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_obs = None

    def _reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        return obs

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        if should_block(self.last_obs, action):
            info["frame/should_have_blocked"] = True
        self.last_obs = obs
        return obs, reward, done, info


if __name__ == "__main__":
    from IPython import get_ipython  # isort:ignore
    from matplotlib import pyplot as plt  # isort:ignore

    ipython = get_ipython()
    ipython.magic("matplotlib inline")

    eps = frame.episode_paths("logs/SpaceInvadersRandom")
    ep = frame.load_episode(eps[0])
    s = set()
    f = np.copy(ep.frames[50].image)
    barrier_damage(f)
    for i in range(0, len(ep.frames)):
        if should_block(ep.frames[i].image, ep.frames[i].action):
            print(i, is_ship_below_barrier(ep.frames[i].image))
            plt.imshow(ep.frames[i].image[155:195, :, :])
            plt.show()
            plt.imshow(ep.frames[i + 1].image[155:195, :, :])
            plt.show()
            plt.imshow(ep.frames[i + 2].image[155:195, :, :])
            plt.show()

    # for i in range(f.shape[0]):
    #     for j in range(f.shape[1]):
    #         if (f[i][j] == np.array([181, 83, 40])).all():
    #             f[i][j] = np.array([255,255,0])
    # for r in ep.frames[0].image:
    #     for c in r:
    #         s.add(tuple(int(x) for x in c))
    # for c in s:
    #     print("rgb{}".format(c))
    # print(s)
    # s.add()
    plt.imshow(f[157:175, :, :])
    plt.show()
