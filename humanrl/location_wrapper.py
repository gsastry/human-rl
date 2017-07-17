import numpy as np
import itertools
import gym
from collections import OrderedDict

def unique_colors(obs, frame_coordinates=None):
    if frame_coordinates is None:
        ranges = map(range, obs.shape[:2])
    else:
        ranges = [range(*r) for r in frame_coordinates]
    
    cache = OrderedDict()
    
    for (row, col) in itertools.product(*ranges):
        color = tuple(obs[row, col].tolist())
        
        if color not in cache:
            cache[color] = (row, col)
    
    return cache

def first_color(obs, color, frame_coordinates=None):
    color = np.array(color)
    
    if frame_coordinates is not None:
        (r1, r2), (c1, c2) = frame_coordinates
        obs = obs[r1:r2, c1:c2]
    
    indices = (obs == color).all(2).nonzero()
    indices = np.array(indices).T
    
    if indices.size:
        return indices[0]
    
    return None

def median_color(obs, color, frame_coordinates=None):
    color = np.array(color)
    
    if frame_coordinates is not None:
        (r1, r2), (c1, c2) = frame_coordinates
        obs = obs[r1:r2, c1:c2]
    
    indices = (obs == color).all(2).nonzero()
    indices = np.array(indices)
    
    if indices.size:
        med = np.median(indices, axis=1)
        return med.astype(np.int32)
    
    return None

def locate_montezuma(obs):
    #red = (232, 204, 99)
    #red = (210, 182, 86)
    #red = (200, 72, 72)
    red = (228, 111, 111)

    return first_color(obs, red)

def locate_space_invaders(obs):
    ship = (50, 132, 50)
    return first_color(obs[30:], ship)

def locate_berzerk(obs):
    guy = (240, 170, 103)
    return median_color(obs, guy)

locators = dict(
    Montezuma=locate_montezuma,
    SpaceInvaders=locate_space_invaders,
    Berzerk=locate_berzerk,
)

class LocationWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LocationWrapper, self).__init__(env)
        
        self.locator = None
        for k, v in locators.items():
            if k in env.spec.id:
                self.locator = v
                break
    
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        if self.locator:
            info['location'] = self.locator(obs)
        
        return obs, reward, done, info

berzerk_coordinates = ((0, 182), (0, 160))

if __name__ == "__main__":
    #env = gym.make("SpaceInvadersDeterministic-v3")
    env = gym.make("BerzerkDeterministic-v3")
    #frame_coordinates = ((32, 160+32), (0, 160))
    #frame_coordinates = ((0, 200), (0, 160))
    frame_coordinates = berzerk_coordinates
    env = LocationWrapper(env)
    env.reset()
    
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        env.render()
        
        cache0 = unique_colors(obs)
        cache1 = unique_colors(obs, frame_coordinates)
        
        #import ipdb; ipdb.set_trace()
        
        #red4 = 
        
        print(info['location'])
        
        if done:
            env.reset()
        #import ipdb; ipdb.set_trace()
