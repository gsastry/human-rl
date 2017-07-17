"""
isort:skip_file
"""

import gym
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def gaussian_kernel(size, std=1.):
    size2 = 1 + 2 * size
    kernel = np.zeros((size2, size2))

    den = 2. * std * std

    for row in range(size2):
        for col in range(size2):
            x = row - size
            y = row - size

            kernel[row, col] = np.exp(-(x*x + y*y) / den)

    kernel /= kernel.sum()

    return kernel

# TODO: check out of bounds
def extract_patch(src, pos, size):
    row, col = pos
    return src[row-size:row+size+1, col-size:col+size+1]

class ExplorationWrapper(gym.Wrapper):
    # assumes env provides location data
    def __init__(self, env, explore_buffer=1e4, bandwidth=3, decay=False, explore_scale=1.0, gamma=0.99, **unused):
        super(ExplorationWrapper, self).__init__(env)
        
        self.explore_scale = explore_scale
        print(explore_scale)

        self.total = int(explore_buffer)

        self.bandwidth = bandwidth
        self.kde = None
        self.locations = []

        self.breadth = int(bandwidth)
        self.kernel = gaussian_kernel(self.breadth, bandwidth)

        rows = 210 + 2 * self.breadth
        cols = 160 + 2 * self.breadth
        self.counts = np.full((rows, cols), self.total / (rows * cols))
        
        self.logprob = -np.log(self.counts.size)

        if decay:
          self.decay = 1. - 1. / self.total
        else:
          self.decay = None
        
        self.gamma = gamma

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)

        location = info.get('location')
        if location is not None:
            """
            self.locations.append(location)
            if len(self.locations) == self.buffer_size:
                # rebuild the kde
                self.kde = stats.gaussian_kde(np.array(self.locations).T, self.bandwidth)

                # plot it?
                dims = obs.shape[:2]
                grid = np.indices(dims)
                kde = self.kde.logpdf(grid.reshape([2, -1]))
                kde = kde.reshape(dims)
                info['kde'] = kde

                #plt.imsave('test.png', kde)

                # drop the older locations
                self.locations = self.locations[self.buffer_size//2:]

            #plt.imsave('counts.png', self.counts)
            #info['logprob'] = logprob

            if self.kde:
                logpdf = self.kde.logpdf(np.array(location))
                info['logpdf'] = logpdf

                reward -= logpdf
            """

            location = location + self.breadth # padding
            index = tuple(location.tolist())

            patch = extract_patch(self.counts, index, self.breadth)
            count = (self.kernel * patch).sum()

            info['log/visits'] = count

            logprob = np.log(count / self.total)
            info['log/visit_logprob'] = logprob

            #reward = 0
            
            bonus = self.explore_scale * (self.logprob - logprob) 
            info['log/explore_bonus'] = np.abs(bonus)
            reward += bonus
            
            self.logprob = logprob

            if self.decay:
                self.counts *= self.decay
            else:
                self.total += 1

            self.counts[index] += 1

        return obs, reward, done, info

if __name__ == "__main__":
    from location_wrapper import LocationWrapper

    env = gym.make("MontezumaRevengeDeterministic-v3")
    env = LocationWrapper(env)
    env = ExplorationWrapper(env)

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        env.render()

        if done:
            env.reset()
