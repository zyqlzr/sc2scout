from gym.spaces import Box

import numpy as np

from sc2scout.wrapper.feature.scout_vec_feature import VecFeature
from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
from sc2scout.wrapper.util.round_trip_status import RoundTripCourse, RoundTripStatus
import sc2scout.envs.scout_macro as sm

class ScoutVecFeatureV3(VecFeature):
    def __init__(self, compress_width, range_width, explore_step):
        super(ScoutVecFeatureV3, self).__init__()
        self._compress_width = compress_width
        self._range_width = range_width
        self._explore_step = explore_step

    def reset(self, env):
        super(ScoutVecFeatureV3, self).reset(env)
        self._init_range_and_status(env)

    def obs_space(self):
        low = np.zeros(12)
        high = np.ones(12)
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

        features.append(scout.float_attr.health / scout.float_attr.health_max)
        if self._have_enemies(obs):
            features.append(float(1))
        else:
            features.append(float(0))

        if self._check_in_target_range(scout):
            features.append(float(1))
        else:
            features.append(float(0))

        self._status.check_status(scout_raw_pos)
        if self._status.status() == RoundTripStatus.EXPLORE:
            #print('vec_feature, explore, pr=', self._status.progress_bar())
            features.append(float(1))
            features.append(float(0))
        else:
            #print('vec_feature, backword, pr=', self._status.progress_bar())
            features.append(float(0))
            features.append(float(1))
        features.append(self._status.progress_bar())
        #print('vec_feature:', features)
        return np.array(features)

    def _have_enemies(self,obs):
        enemy_count = 0
        units = obs.observation['units']
        for u in units:
            if (u.int_attr.alliance == sm.AllianceType.ENEMY.value):
               if (u.unit_type in sm.COMBAT_UNITS or u.unit_type in sm.COMBAT_AIR_UNITS):
                enemy_count += 1
        if enemy_count > 0:
            return True
        else:
            return False

    def _init_range_and_status(self, env):
        home = env.unwrapped.owner_base()
        target = env.unwrapped.enemy_base()
        map_size = env.unwrapped.map_size()
        x_per_unit = map_size[0] / self._compress_width
        y_per_unit = map_size[1] / self._compress_width

        x_range = x_per_unit * self._range_width
        y_range = y_per_unit * self._range_width
        x_radius = x_range / 2
        y_radius = y_range / 2
        self._x_low = target[0] - x_radius
        self._x_high = target[0] + x_radius
        self._y_low = target[1] - y_radius
        self._y_high = target[1] + y_radius
        print('Evade per_unit=({},{}), radius=({},{}), x_range=({},{}), y_range=({},{}), target={}'.format(
            x_per_unit, y_per_unit, x_radius, y_radius,
            self._x_low, self._x_high, self._y_low, self._y_high, target))
        self._status = RoundTripCourse(home, target, (x_range, y_range), self._explore_step)
        self._status.reset()

    def _check_in_target_range(self, scout):
        if scout.float_attr.pos_x > self._x_high:
            return False
        elif scout.float_attr.pos_x < self._x_low:
            return False
        elif scout.float_attr.pos_y > self._y_high:
            return False
        elif scout.float_attr.pos_y < self._y_low:
            return False
        else:
            return True


