import gym
import numpy as np
from gym.spaces import Box


class ZergScoutObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ZergScoutObsWrapper, self).__init__(env)
        self._init_obs_space()
        self._map_size = self.env.unwrapped.map_size()

    def _reset(self):
        obs = self.env._reset()
        obs = self.observation(obs)
        print('obswrapper map_size={}'.format(self._map_size))
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    '''
    [scout_x,
     scout_y,
     enemy_x,
     enemy_y,
     scout_health
    ]
    '''
    def _init_obs_space(self):
        low = np.zeros(5)
        high = np.ones(5)
        self.observation_space = Box(low, high)

    def _observation(self, obs):
        scout = self.env.unwrapped.scout()
        enemy = self.env.unwrapped.enemy()
        view_enemy = self.env.unwrapped.view_enemy()
        if view_enemy:
            return np.array([scout.float_attr.pos_x / self._map_size[0],
                         scout.float_attr.pos_y / self._map_size[1],
                         enemy.float_attr.pos_x / self._map_size[0],
                         enemy.float_attr.pos_y / self._map_size[1],
                         scout.float_attr.health / scout.float_attr.health_max])
        else:# enemy is not in the viewing circle
            return np.array([scout.float_attr.pos_x / self._map_size[0],
                             scout.float_attr.pos_y / self._map_size[1],
                             1,
                             1,
                             scout.float_attr.health / scout.float_attr.health_max])




