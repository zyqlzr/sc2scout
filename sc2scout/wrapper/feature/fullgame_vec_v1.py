from gym.spaces import Box
import numpy as np

from pysc2.lib.typeenums import RACE, UNIT_TYPEID, ABILITY_ID, UPGRADE_ID, BUFF_ID

from sc2scout.wrapper.feature.scout_vec_feature import VecFeature
from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
from sc2scout.wrapper.util.dest_range import DestRange
import sc2scout.envs.scout_macro as sm

MAX_RADIUS_NUM = 360.
MAX_VIEW_RANGE = 13.
MAX_UNIT_NUM = 200.
MAX_BASE_NUM = 16.

class FullGameVecV1(VecFeature):
    def __init__(self, target_range):
        super(FullGameVecV1, self).__init__()
        self._base_num = 0
        self._force_num = 0
        self._other_num = 0
        self._building_type_num = 0
        self._target_range = target_range
        self._target = None
        self._unit_set = set([])
        self._building_type_set = set([])
        self._base_id_set = set([])

    def reset(self, env):
        super(FullGameVecV1, self).reset(env)
        self._target = DestRange(env.unwrapped.enemy_base(), 
                                 dest_range=self._target_range)
        self._base_num = 0
        self._force_num = 0
        self._other_num = 0
        self._building_type_num = 0
        self._unit_set = set([])
        self._building_type_set = set([])
        self._base_id_set = set([])

    def obs_space(self):
        low = np.zeros(14)
        high = np.ones(14)
        return Box(low, high)

    def extract(self, env, obs):
        features = []
        enemys = self._unit_dispatch(obs)
        features = self._normal_feature(features, env)
        features = self._status_feature(features, env)
        features = self._count_feature(features, enemys, env)
        return features

    def _normal_feature(self, features, env):
        scout = env.unwrapped.scout()
        scout_raw_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        scout_pos = self._pos_transfer(scout_raw_pos[0], scout_raw_pos[1])
        home_pos = self._pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self._pos_transfer(enemy_pos[0], enemy_pos[1])

        features.append(float(scout_pos[0]) / self._map_size[0])
        features.append(float(scout_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(scout.float_attr.facing / MAX_RADIUS_NUM)
        features.append(scout.float_attr.health / scout.float_attr.health_max)
        features.append(float(env.unwrapped.step_number() / env.unwrapped.max_step_number()))
        return features

    def _status_feature(self, features, env):
        scout = env.unwrapped.scout()
        if self._target.in_range((scout.float_attr.pos_x,
                                  scout.float_attr.pos_y)):
            features.append(float(1))
        else:
            features.append(float(0))
        return features

    def _count_feature(self, features, enemys, env):
        for u in enemys:
            if not self._check_view(env, u):
                continue

            if u.unit_type in sm.BASE_UNITS:
                bid = env.unwrapped.get_id_by_pos((u.float_attr.pos_x, 
                                                   u.float_attr.pos_y))
                #print('find_bid=', bid)
                if bid in self._base_id_set:
                    pass
                else:
                    self._base_num += 1
                    self._base_id_set.add(bid)
            elif u.unit_type in sm.BUILDING_UNITS:
                if u.unit_type in self._building_type_set:
                    pass
                else:
                    self._building_type_num += 1
                    self._building_type_set.add(u.unit_type)
            elif u.unit_type in sm.COMBAT_ATTACK_UNITS:
                if u.tag in self._unit_set:
                    pass
                else:
                    self._force_num += 1
                    self._unit_set.add(u.tag)
            else:
                if u.tag in self._unit_set:
                    pass
                else:
                    self._other_num += 1
                    self._unit_set.add(u.tag)

        features.append(float(self._base_num / MAX_BASE_NUM))
        features.append(float(self._building_type_num / MAX_UNIT_NUM))
        features.append(float(self._force_num / MAX_UNIT_NUM))
        features.append(float(self._other_num / MAX_UNIT_NUM))
        #print('FULLGAME_VEC base={},building={},force={},other={}'.format(
        #      self._base_num, self._building_type_num,
        #      self._force_num, self._other_num))
        return features

    def _check_view(self, env, unit):
        scout = env.unwrapped.scout()
        dist = sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     unit.float_attr.pos_x,
                                     unit.float_attr.pos_y)
        if dist <= MAX_VIEW_RANGE:
            return True
        else:
            return False

    def _unit_dispatch(self, obs):
        units = obs.observation['units']
        enemys = []
        for u in units:
            if u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        return enemys


