from gym.spaces import Box
import numpy as np

from sc2scout.wrapper.feature.scout_vec_feature import VecFeature
from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
import sc2scout.envs.scout_macro as sm

MAX_RADIUS_NUM = 360.

class FullGameVec(VecFeature):
    def __init__(self):
        super(FullGameVec, self).__init__()

    def reset(self, env):
        super(FullGameVec, self).reset(env)

    def obs_space(self):
        low = np.zeros(8)
        high = np.ones(8)
        return Box(low, high)

    def extract(self, env, obs):
        scout = env.unwrapped.scout()
        scout_raw_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        scout_pos = self._pos_transfer(scout_raw_pos[0], scout_raw_pos[1])
        home_pos = self._pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self._pos_transfer(enemy_pos[0], enemy_pos[1])

        features = []
        features.append(float(scout_pos[0]) / self._map_size[0])
        features.append(float(scout_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(scout.float_attr.facing / MAX_RADIUS_NUM)
        features.append(scout.float_attr.health / scout.float_attr.health_max)

        #print("vecs=", features)
        return features


