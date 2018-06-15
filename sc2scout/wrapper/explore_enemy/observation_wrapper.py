import gym
import numpy as np
from gym.spaces import Box

SIMPLE_MAP_MAX_X = 88
SIMPLE_MAX_MAX_Y = 96

class ZergScoutObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ZergScoutObsWrapper, self).__init__(env)
        self._init_obs_space()

    def _reset(self):
        obs = self.env._reset()
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    '''
    [scout_x,
     scout_y,
     home_x,
     home_y,
     enemy_base_x,
     enemy_base_y
    ]
    '''
    def _init_obs_space(self):
        low = np.zeros(6)
        high = np.ones(6)
        self.observation_space = Box(low, high)

    def _observation(self, obs):
        scout = self.env.unwrapped.scout()
        scout_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = self.env.unwrapped.owner_base()
        enemy_pos = self.env.unwrapped.enemy_base()
        return np.array([float(scout_pos[0]) / SIMPLE_MAP_MAX_X,
                         float(scout_pos[1]) / SIMPLE_MAX_MAX_Y,
                         float(home_pos[0]) / SIMPLE_MAP_MAX_X,
                         float(home_pos[1]) / SIMPLE_MAX_MAX_Y,
                         float(enemy_pos[0]) / SIMPLE_MAP_MAX_X, 
                         float(enemy_pos[1]) / SIMPLE_MAX_MAX_Y])


