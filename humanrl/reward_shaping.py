import gym


class RewardShaper(gym.Wrapper):
    def __init__(self, env, gamma=0.99):
        super(RewardShaper, self).__init__(env)
        self.gamma = gamma
        self.prev_potential = 0

    def potential(self, state):
        return 0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        potential = self.potential(obs)
        reward += self.gamma * potential - self.prev_potential
        self.prev_potential = potential
        return obs, reward, done, info


class HumanShaper(RewardShaper):
    def __init__(self, env, **kwargs):
        super(HumanShaper, self).__init__(env, **kwargs)

        self.good = False
        self.bad = False

        env.render()
        env.viewer.window.on_key_press = self.key_press
        env.viewer.window.on_key_release = self.key_release

    def key_press(self, key, mod):
        if key == ord('1'):
            self.good = True
        elif key == ord('2'):
            self.bad = True

    def key_release(self, key, mod):
        if key == ord('1'):
            self.good = False
        elif key == ord('2'):
            self.bad = False

    def potential(self, _state):
        v = 0
        if self.good:
            v += 1
        if self.bad:
            v -= 1
        return v
