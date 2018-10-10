from gym.spaces import Box
import numpy as np

from pysc2.lib.typeenums import RACE, UNIT_TYPEID, ABILITY_ID, UPGRADE_ID, BUFF_ID

from sc2scout.wrapper.feature.scout_vec_feature import VecFeature
from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
import sc2scout.envs.scout_macro as sm

MAX_RADIUS_NUM = 360.
MAX_VIEW_RANGE = 13

QUEEN_TYPES = set([
    UNIT_TYPEID.ZERG_QUEEN.value,
    UNIT_TYPEID.ZERG_QUEENBURROWED.value
])

BASE_UPGRADE = set([
    UNIT_TYPEID.ZERG_LAIR.value,
    UNIT_TYPEID.ZERG_HIVE.value
])

class FullGameVecAll(VecFeature):
    def __init__(self):
        super(FullGameVecAll, self).__init__()
        self._spawning_pool = False
        self._base_upgrade = False
        self._air_force = False
        self._land_force = False

    def reset(self, env):
        super(FullGameVecAll, self).reset(env)

    def obs_space(self):
        low = np.zeros(13)
        high = np.ones(13)
        return Box(low, high)

    def extract(self, env, obs):
        self._analysis_enemy(obs, env)

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
        features.append(float(1 if self._spawning_pool else 0))
        features.append(float(1 if self._air_force else 0))
        features.append(float(1 if self._land_force else 0))
        features.append(float(1 if self._base_upgrade else 0))
        features.append(float(env.unwrapped.step_number() / env.unwrapped.max_step_number()))
        #print("vecs=", features)
        return features

    def _analysis_enemy(self, obs, env):
        enemys = self._unit_dispatch(obs)
        for u in enemys:
            if u.unit_type == UNIT_TYPEID.ZERG_SPAWNINGPOOL.value:
                if not self._spawning_pool:
                    self._spawning_pool = True
            elif u.unit_type in QUEEN_TYPES or u.unit_type in sm.COMBAT_AIR_UNITS:
                if not self._air_force:
                    self._air_force = True
            elif u.unit_type in sm.COMBAT_UNITS:
                if not self._land_force:
                    self._land_force = True
            elif u.unit_type in BASE_UPGRADE:
                if not self._base_upgrade:
                    self._base_upgrade = True
            else:
                pass

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


