import gym


# TODO - "max_reward" as a name make sense?
class EarlyTerminationWrapper(gym.Wrapper):
    """
    Wrapper that determines when to terminate the episode.

    This currently just supports termination due to a reward cutoff,
    but can be extended to take in a termination function in the future.

    """
    def __init__(self, env, max_episode_penalty=None):
        super(EarlyTerminationWrapper, self).__init__(env)

        self.max_episode_penalty = max_episode_penalty
        self.episode_reward = 0
        self.lost_reward = 0  # todo(girish): hack to run this overnight

    @property
    def _get_final_reward(self):
        return self.max_episode_penalty

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self.max_episode_penalty is not None:
            self.episode_reward += reward

            if reward < 0:
                self.lost_reward += abs(reward)

            if self.lost_reward >= abs(self.max_episode_penalty):
                done = True
                self.episode_reward = 0
                self.lost_reward = 0

        return obs, reward, done, info
