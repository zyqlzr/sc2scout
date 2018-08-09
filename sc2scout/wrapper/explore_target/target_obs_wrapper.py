import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.scout_global_img_feature_v1 import ScoutGlobalImgFeatureV1

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

