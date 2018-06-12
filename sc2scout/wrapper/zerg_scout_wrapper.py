import gym
from sc2scout.envs import SC2GymEnv
import numpy as np
from gym.spaces import Box

class ZergScoutWrapper(gym.Wrapper):
    def __init__(self, env):
        print('ZergScoutWrapper initilize')
        super(ZergScoutWrapper, self).__init__(env)
        print('act_space_shape=', self.action_space.shape)
        print('obs_space_shape=', self.observation_space.shape)

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        return self.env._step(action)
