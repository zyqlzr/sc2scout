import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.fullgame_global_img import FullGameGlobalImg
from sc2scout.wrapper.feature.fullgame_vec import FullGameVec
from sc2scout.wrapper.feature.fullgame_vec_minimap import FullGameVecMinimap

class FullGameObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width):
        super(FullGameObsWrapper, self).__init__(env)
        self._global = FullGameGlobalImg(compress_width, range_width)
        self._vec = FullGameVec()
        self._init_obs_space()
        print("FullGameObsWrapper: g_shape={};v_shape={};total_obs_shape={}".format(
              self._global.obs_space().shape, self._vec.obs_space().shape,
              self.observation_space.shape))

    def _reset(self):
        obs = self.env._reset()
        self._global.reset(self.env)
        self._vec.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def observation(self, obs):
        g_img = self._global.extract(self.env, obs)
        vec = self._vec.extract(self.env, obs)
        return np.hstack([g_img.flatten(), vec])

    def _get_dim(self, ob_space):
        shape_size = len(ob_space.shape)
        dim = 1
        for i in range(0, shape_size):
            dim = dim * ob_space.shape[i]
        return dim

    def _init_obs_space(self):
        g_dim = self._get_dim(self._global.obs_space())
        v_dim = self._get_dim(self._vec.obs_space())
        low =  np.zeros(g_dim + v_dim)
        high = np.ones(g_dim + v_dim)
        self.observation_space = Box(low, high)
        print("fullgame obs space:", self.observation_space)


class FullGameObsMiniWrapper(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width):
        super(FullGameObsMiniWrapper, self).__init__(env)
        self._global = FullGameGlobalImg(compress_width, range_width)
        self._vec = FullGameVecMinimap()
        self._init_obs_space()
        print("FullGameObsWrapper: g_shape={};v_shape={};total_obs_shape={}".format(
              self._global.obs_space().shape, self._vec.obs_space().shape,
              self.observation_space.shape))

    def _reset(self):
        obs = self.env._reset()
        self._global.reset(self.env)
        self._vec.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def observation(self, obs):
        g_img = self._global.extract(self.env, obs)
        vec = self._vec.extract(self.env, obs)
        return np.hstack([g_img.flatten(), vec])

    def _get_dim(self, ob_space):
        shape_size = len(ob_space.shape)
        dim = 1
        for i in range(0, shape_size):
            dim = dim * ob_space.shape[i]
        return dim

    def _init_obs_space(self):
        g_dim = self._get_dim(self._global.obs_space())
        v_dim = self._get_dim(self._vec.obs_space())
        low =  np.zeros(g_dim + v_dim)
        high = np.ones(g_dim + v_dim)
        self.observation_space = Box(low, high)
        print("fullgame_mini obs space:", self.observation_space)


