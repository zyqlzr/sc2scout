from gym.spaces import Box

import numpy as np

from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
from sc2scout.wrapper.util.dest_range import DestRange
import sc2scout.envs.scout_macro as sm

SCOUT_IN_RANGE = 1
SCOUT_OUT_RANGE = 0

class VecFeature(FeatureExtractor):
    def __init__(self):
        self._reverse = False
        self._map_size = None

    def reset(self, env):
        self._map_size = env.unwrapped.map_size()
        self._reverse = self._judge_reverse(env)

    def _judge_reverse(self, env):
        scout = env.unwrapped.scout()
        if scout.float_attr.pos_x < scout.float_attr.pos_y:
            return False
        else:
            return True

    def _pos_transfer(self, x, y):
        if not self._reverse:
            return (x, y)
        cx = self._map_size[0] / 2
        cy = self._map_size[1] / 2
        pos_x = 0.0
        pos_y = 0.0
        if x > cx:
            pos_x = cx - abs(x - cx)
        else:
            pos_x = cx + abs(x - cx)

        if y > cy:
            pos_y = cy - abs(y - cy)
        else:
            pos_y = cy + abs(y - cy)

        return (pos_x, pos_y)

class ScoutSimpleFeature(VecFeature):
    def __init__(self):
        super(ScoutSimpleFeature, self).__init__()

    def reset(self, env):
        super(ScoutSimpleFeature, self).reset(env)

    def obs_space(self):
        low = np.zeros(6)
        high = np.ones(6)
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
        return features

class ScoutVecFeature(VecFeature):
    def __init__(self):
        super(ScoutVecFeature, self).__init__()
        self._dest = None
        self._src = None

    def reset(self, env):
        super(ScoutVecFeature, self).reset(env)
        self._dest = DestRange(env.unwrapped.enemy_base())
        self._src = DestRange(env.unwrapped.owner_base())

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
        #features.append(float(home_pos[0]) / self._map_size[0])
        #features.append(float(home_pos[1]) / self._map_size[1])
        #features.append(float(enemy_pos[0]) / self._map_size[0])
        #features.append(float(enemy_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])

        if self._dest.in_range(scout_raw_pos):
            features.append(float(1))
        else:
            features.append(float(0))

        if self._src.in_range(scout_raw_pos):
            features.append(float(1))
        else:
            features.append(float(0))

        return features

class ScoutVecFeatureV1(VecFeature):
    def __init__(self):
        super(ScoutVecFeatureV1, self).__init__()
        self._dest = None
        self._src = None

    def reset(self, env):
        super(ScoutVecFeatureV1, self).reset(env)
        self._dest = DestRange(env.unwrapped.enemy_base())
        self._src = DestRange(env.unwrapped.owner_base())

    def obs_space(self):
        low = np.zeros(4)
        high = np.ones(4)
        return Box(low, high)

    def extract(self, env, obs):
        scout = env.unwrapped.scout()

        features = []
        features.append(scout.float_attr.health / scout.float_attr.health_max)
        if self._have_enemies(obs):
            features.append(float(1))
        else:
            features.append(float(0))

        if self._src.in_range((scout.float_attr.pos_x, scout.float_attr.pos_y)):
            features.append(float(1))
        else:
            features.append(float(0))

        if self._dest.in_range((scout.float_attr.pos_x, scout.float_attr.pos_y)):
            features.append(float(1))
        else:
            features.append(float(0))

        #print("vec_feature=", features)
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


class ScoutVecFeatureV2(VecFeature):
    def __init__(self, compress_width, range_width):
        super(ScoutVecFeatureV2, self).__init__()
        self._compress_width = compress_width
        self._range_width = range_width

    def reset(self, env):
        super(ScoutVecFeatureV2, self).reset(env)
        self._compute_target_range(env)

    def obs_space(self):
        low = np.zeros(9)
        high = np.ones(9)
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

    def _compute_target_range(self, env):
        target = env.unwrapped.enemy_base()
        map_size = env.unwrapped.map_size()
        x_per_unit = map_size[0] / self._compress_width
        y_per_unit = map_size[1] / self._compress_width
        x_radius = (x_per_unit * self._range_width) / 2
        y_radius = (y_per_unit * self._range_width) / 2
        self._x_low = target[0] - x_radius
        self._x_high = target[0] + x_radius
        self._y_low = target[1] - y_radius
        self._y_high = target[1] + y_radius
        print('Evade per_unit=({},{}), radius=({},{}), x_range=({},{}), y_range=({},{}), target={}'.format(
            x_per_unit, y_per_unit, x_radius, y_radius,
            self._x_low, self._x_high, self._y_low, self._y_high, target))

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


