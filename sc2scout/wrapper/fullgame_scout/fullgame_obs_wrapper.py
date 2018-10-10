import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.fullgame_global_img import FullGameGlobalImg
from sc2scout.wrapper.feature.fullgame_global_img_v1 import FullGameGlobalImgV1
from sc2scout.wrapper.feature.fullgame_vec import FullGameVec
from sc2scout.wrapper.feature.fullgame_vec_minimap import FullGameVecMinimap
from sc2scout.wrapper.feature.fullgame_local_img import FullGameLocalImg
from sc2scout.wrapper.feature.fullgame_vec_all import FullGameVecAll

class FullGameObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, compress_width):
        super(FullGameObsWrapper, self).__init__(env)
        self._global = FullGameGlobalImg(compress_width)
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
    def __init__(self, env, compress_width):
        super(FullGameObsMiniWrapper, self).__init__(env)
        self._global = FullGameGlobalImg(compress_width)
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


class FullGameObsMiniWrapperV1(gym.ObservationWrapper):
    def __init__(self, env, compress_width):
        super(FullGameObsMiniWrapperV1, self).__init__(env)
        self._global = FullGameGlobalImgV1(compress_width)
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


class FullGameObsMiniWrapperV2(gym.ObservationWrapper):
    def __init__(self, env, compress_width, local_range):
        super(FullGameObsMiniWrapperV2, self).__init__(env)
        self._global = FullGameGlobalImgV1(compress_width)
        self._local = FullGameLocalImg(compress_width, local_range)
        self._vec = FullGameVecMinimap()
        self._init_obs_space()
        print("TargetObsWrapperV3: g_shape={};l_shape={};v_shape={};total_obs_shape={}".format(
              self._global.obs_space().shape, self._local.obs_space().shape,
              self._vec.obs_space().shape, self.observation_space.shape))

    def _reset(self):
        obs = self.env._reset()
        self._global.reset(self.env)
        self._local.reset(self.env)
        self._vec.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def observation(self, obs):
        g_img = self._global.extract(self.env, obs)
        l_img = self._local.extract(self.env, obs)
        vec = self._vec.extract(self.env, obs)
        return np.hstack([g_img.flatten(), l_img.flatten(), vec])

    def _get_dim(self, ob_space):
        shape_size = len(ob_space.shape)
        dim = 1
        for i in range(0, shape_size):
            dim = dim * ob_space.shape[i]
        return dim

    def _init_obs_space(self):
        g_dim = self._get_dim(self._global.obs_space())
        l_dim = self._get_dim(self._local.obs_space())
        v_dim = self._get_dim(self._vec.obs_space())
        low =  np.zeros(g_dim + l_dim + v_dim)
        high = np.ones(g_dim + l_dim + v_dim)
        self.observation_space = Box(low, high)
        print("obs space", self.observation_space)


class FullGameObsWrapperV1(gym.ObservationWrapper):
    def __init__(self, env, compress_width, local_range):
        super(FullGameObsWrapperV1, self).__init__(env)
        self._global = FullGameGlobalImgV1(compress_width)
        self._local = FullGameLocalImg(compress_width, local_range)
        self._vec = FullGameVecAll()
        self._init_obs_space()
        print("FullGameObsWrapperV1: g_shape={};l_shape={};v_shape={};total_obs_shape={}".format(
              self._global.obs_space().shape, self._local.obs_space().shape,
              self._vec.obs_space().shape, self.observation_space.shape))

    def _reset(self):
        obs = self.env._reset()
        self._global.reset(self.env)
        self._local.reset(self.env)
        self._vec.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def observation(self, obs):
        g_img = self._global.extract(self.env, obs)
        l_img = self._local.extract(self.env, obs)
        vec = self._vec.extract(self.env, obs)
        return np.hstack([g_img.flatten(), l_img.flatten(), vec])

    def _get_dim(self, ob_space):
        shape_size = len(ob_space.shape)
        dim = 1
        for i in range(0, shape_size):
            dim = dim * ob_space.shape[i]
        return dim

    def _init_obs_space(self):
        g_dim = self._get_dim(self._global.obs_space())
        l_dim = self._get_dim(self._local.obs_space())
        v_dim = self._get_dim(self._vec.obs_space())
        low =  np.zeros(g_dim + l_dim + v_dim)
        high = np.ones(g_dim + l_dim + v_dim)
        self.observation_space = Box(low, high)
        print("obs space", self.observation_space)

