import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.scout_global_img_feature_v1 import ScoutGlobalImgFeatureV1
from sc2scout.wrapper.feature.scout_vec_feature import ScoutStaticsticVec

class TargetObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(TargetObsWrapper, self).__init__(env)
        self._obs = (
            ScoutGlobalImgFeatureV1(compress_width, range_width, #global observation
                                    #local observation
            explore_step, False),ScoutStaticsticVec(compress_width,range_width,explore_step,False) #vector observation
        )
        self._init_obs_space()

    def _reset(self):
        obs = self.env._reset()
        self._obs[0].reset(self.env)
        self._obs[1].reset(self.env)
        self._obs[2].reset(self.env)
        obs = self.observation(obs,action=0)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs,action)
        return obs, rwd, done, other

    def _init_obs_space(self):
        vec_dim = self._get_dim(self._obs[0].obs_space()) \
                  + self._get_dim(self._obs[1].obs_space()) \
                  + self._obs[2].obs_space().shape[0]
        low =  np.zeros(vec_dim)
        high = np.ones(vec_dim)
        self.observation_space = Box(low, high)
        print("obs space", self.observation_space)

    def observation(self, obs, action):
        global_img_features = self._obs[0].extract(self.env, obs)
        local_img_features = self._obs[1].extract(self.env, obs)
        vec_features = self._obs[2].extract(self.env,obs,action)
        # print('statistic vec', vec_features)
        global_features_flatten = global_img_features.flatten()
        local_features_flatten = local_img_features.flatten(())
        return np.hstack([global_features_flatten, local_features_flatten ,vec_features])

    def _get_dim(self,box):
        return box.shape[0] * box.shape[1] * box.shape[2]
