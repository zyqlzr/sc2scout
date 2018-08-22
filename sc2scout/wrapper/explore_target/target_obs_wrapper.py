import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.scout_global_img_feature import ScoutGlobalImgFeature
from sc2scout.wrapper.feature.scout_global_img_feature_v1 import ScoutGlobalImgFeatureV1
from sc2scout.wrapper.feature.scout_global_img_feature_v2 import ScoutGlobalImgFeatureV2
from sc2scout.wrapper.feature.scout_local_img_feature_v1 import ScoutLocalImgFeatureV1
from sc2scout.wrapper.feature.scout_local_img_feature_v2 import ScoutLocalImgFeatureV2
from sc2scout.wrapper.feature.scout_vec_feature import ScoutVecFeatureV1, ScoutVecFeatureV2
from sc2scout.wrapper.feature.scout_global_img_feature_v3 import ScoutGlobalImgFeatureV3

class TargetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(TargetObsWrapper, self).__init__(env)
        self._obs = ScoutGlobalImgFeatureV1(compress_width, range_width, explore_step, False)
        self._init_obs_space()

    def _reset(self):
        obs = self.env._reset()
        self._obs.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def _init_obs_space(self):
        self.observation_space = self._obs.obs_space()
        print('Evade img obs space=', self._obs.obs_space())

    def _observation(self, obs):
        return self._obs.extract(self.env, obs)


class TargetObsWrapperV1(gym.ObservationWrapper):
    def __init__(self, env, global_compress_width, local_compress_width, 
                 local_range_width, reverse):
        super(TargetObsWrapperV1, self).__init__(env)
        self._global = ScoutGlobalImgFeatureV2(global_compress_width, False)
        self._local = ScoutLocalImgFeatureV1(local_compress_width,
                                             local_range_width, False)
        self._vec = ScoutVecFeatureV1()
        self._init_obs_space()
        print("obs_wrapper ,g_shape={},l_shape={},v_shape={};total_obs_shape={}".format(
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

class TargetObsWrapperV2(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(TargetObsWrapperV2, self).__init__(env)
        self._obs = ScoutGlobalImgFeatureV3(compress_width, range_width, explore_step, False)
        self._init_obs_space()

    def _reset(self):
        obs = self.env._reset()
        self._obs.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def _init_obs_space(self):
        self.observation_space = self._obs.obs_space()
        print('Evade img obs space=', self._obs.obs_space())

    def _observation(self, obs):
        return self._obs.extract(self.env, obs)

class TargetObsWrapperV3(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width, reverse):
        super(TargetObsWrapperV3, self).__init__(env)
        self._global = ScoutGlobalImgFeature(compress_width, False)
        self._vec = ScoutVecFeatureV2(compress_width, range_width)
        self._init_obs_space()
        print("TargetObsWrapperV3: g_shape={};v_shape={};total_obs_shape={}".format(
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
        print("obs space", self.observation_space)

class TargetObsWrapperV4(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width,
                 local_compress_width, local_range_width, reverse):
        super(TargetObsWrapperV4, self).__init__(env)
        self._global = ScoutGlobalImgFeature(compress_width, False)
        self._local = ScoutLocalImgFeatureV2(local_compress_width, local_range_width, False)
        self._vec = ScoutVecFeatureV2(compress_width, range_width)
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


