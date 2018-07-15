import gym
import numpy as np
from gym.spaces import Box

from sc2scout.wrapper.feature.scout_global_img_feature import ScoutGlobalImgFeature
from sc2scout.wrapper.feature.scout_local_img_feature import ScoutLocalImgFeature


class ScoutEvadeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScoutEvadeObsWrapper, self).__init__(env)
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


class ScoutEvadeBlendObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScoutEvadeBlendObsWrapper, self).__init__(env)
        self._init_obs_space()
        self._global = ScoutGlobalImgFeature(32, False)
        self._local = ScoutLocalImgFeature(32, 12, False)
        self._vec = None

    def _reset(self):
        obs = self.env._reset()
        self._local.reset(self.env)
        self._global.reset(self.env)

        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def _init_obs_space(self):
        low = np.zeros(5)
        high = np.ones(5)
        self.observation_space = Box(low, high)

    def _observation(self, obs):
        self._local.extract(self.env, obs)
        self._global.extract(self.env, obs)
        return obs
