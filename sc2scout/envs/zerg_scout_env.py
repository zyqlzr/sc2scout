import gym
from sc2scout.envs.sc2_gym_env import SC2GymEnv
from sc2scout.envs import scout_macro as sm
from pysc2.lib.typeenums import UNIT_TYPEID
import numpy as np

RESOURCE_DISTANCE = 7.0

class ZergScoutEnv(SC2GymEnv):
    def __init__(self, **kwargs):
        super(ZergScoutEnv, self).__init__(**kwargs)
        self._init_action_space()
        self._scout = None
        self._owner_base_pos = None
        self._enemy_base_pos = None
        self._base_candidates = []

    def _init_action_space(self):
        self.action_space = gym.spaces.Discrete(8)

    def _reset(self):
        obs = super(ZergScoutEnv, self)._reset()
        self._init_scout_and_base(obs)
        print('*** ZergScoutEnv scout_unit=', self._scout.tag)
        print('*** ZergScoutEnv base_candidates=', self._base_candidates)
        return obs

    def base_candidates(self):
        return self._base_candidates

    def scout(self):
        return self._scout

    def _init_scout_and_base(self, obs):
        units = obs.observation['units']

        tmps = self._unit_dispatch(units)
        my_base = tmps[0][0]
        self._owner_base_pos = (my_base.float_attr.pos_x, my_base.float_attr.pos_y)
        print('owner_base=', self._owner_base_pos)

        minerals = {}
        vespenes = {}

        for unit in tmps[1]:
            minerals[unit.tag] = unit

        for unit in tmps[2]:
            vespenes[unit.tag] = unit

        mtags = set(minerals.keys())
        gtags = set(vespenes.keys())
        while len(mtags) > 0 and len(gtags) > 0:
            # print('mtags_num=', len(mtags), ';gtags_num=', len(gtags))
            pos = self._find_resource_area(mtags, gtags,
                                           minerals, vespenes)
            self._base_candidates.append(pos)
        self._find_enemy_base_from_candidates()
        print('enemy_base=', self._enemy_base_pos)

    def _find_enemy_base_from_candidates(self):
        max_pos = None
        max_dist = None
        for pos in self._base_candidates:
            tmp_dist = self._calculate_distances(pos[0], pos[1],
                                      self._owner_base_pos[0],
                                      self._owner_base_pos[1])
            if max_dist is None:
                max_dist = tmp_dist
                max_pos = pos
            elif max_dist < tmp_dist:
                max_dist = tmp_dist
                max_pos = pos
            else:
                pass
        self._enemy_base_pos = max_pos

    def _unit_dispatch(self, units):
        tmp_base = []
        tmp_minerals = []
        tmp_vespene = []
        for u in units:
            if self._check_base(u):
                tmp_base.append(u)
            elif self._check_mineral(u):
                tmp_minerals.append(u)
            elif self._check_vespene(u):
                tmp_vespene.append(u)
            elif u.int_attr.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value:
                if self._scout is None:
                    self._scout = u
                    print('find scout,unit=', self._scout.tag, ',unit_pos=',
                          (self._scout.float_attr.pos_x, self._scout.float_attr.pos_y))
            else:
                pass  # do nothing

        return (tmp_base, tmp_minerals, tmp_vespene)

    def _find_resource_area(self, mtags, gtags, all_minerals, all_gas):
        gtags_in_area = []
        tag = mtags.pop()
        mtags_in_area = [tag]
        while mtags:  # not empty
            d_min = None
            mtag = None
            for tag in mtags:
                d = self._min_dist(all_minerals[tag], mtags_in_area,
                                  gtags_in_area, all_minerals, all_gas)
                if d_min is None or d < d_min:
                    d_min = d
                    mtag = tag
            if d_min > RESOURCE_DISTANCE:
                break
            mtags_in_area.append(mtag)
            mtags.discard(mtag)

        while gtags:  # not empty
            d_min = None
            gtag = None
            for tag in gtags:
                d = self._min_dist(all_gas[tag], mtags_in_area,
                                  gtags_in_area, all_minerals, all_gas)
                if d_min is None or d < d_min:
                    d_min = d
                    gtag = tag
            if d_min > RESOURCE_DISTANCE:
                break
            gtags_in_area.append(gtag)
            gtags.discard(gtag)

        m_pos = [[all_minerals[tag].float_attr.pos_x,
                  all_minerals[tag].float_attr.pos_y] for tag in mtags_in_area]
        g_pos = [[all_gas[tag].float_attr.pos_x,
                  all_gas[tag].float_attr.pos_y] for tag in gtags_in_area]
        ideal_pos = self._find_ideal_base_position(np.array(m_pos),
                                                  np.array(g_pos))
        return ideal_pos

    def _min_dist(self, unit, mtags, gtags, all_minerals, all_gas):
        # minimal dist from unit to mtags and gtags
        d = [self._unit_dist(unit, all_minerals[tag]) for tag in mtags] + \
            [self._unit_dist(unit, all_gas[tag]) for tag in gtags]
        return min(d)

    def _find_ideal_base_position(self, m_pos, g_pos):
        mean_x, mean_y = m_pos.mean(0)
        max_x, max_y = g_pos.min(0) + 10
        min_x, min_y = g_pos.max(0) - 10
        d_min = None
        ideal_pos = []
        x = min_x
        while x <= max_x:
            y = min_y
            while y <= max_y:
                if self._can_build_base(x, y, m_pos):
                    d = self._calculate_distances(x, y, mean_x, mean_y)
                    if d_min is None or d < d_min:
                        ideal_pos = [x, y]
                        d_min = d
                y += 1
            x += 1
        return ideal_pos

    def _calculate_distances(self, x1, y1, x2, y2):
        x = abs(x1 - x2)
        y = abs(y1 - y2)
        distance = x ** 2 + y ** 2
        return distance ** 0.5

    def _unit_dist(self, unit1, unit2):
        return self._calculate_distances(unit1.float_attr.pos_x,
                                        unit1.float_attr.pos_y,
                                        unit2.float_attr.pos_x,
                                        unit2.float_attr.pos_y)


    def _can_build_base(self, x, y, m_pos):
        for pos in m_pos:
            dx = abs(pos[0] - x)
            dy = abs(pos[1] - y)
            if dx < 6 and dy < 6 and (dx < 5 or dy < 5):
                return False
        return True

    def _check_base(self, u):
        if u.unit_type in sm.BASE_UNITS and \
                        u.int_attr.alliance == sm.AllianceType.SELF.value:
            return True
        else:
            return False

    def _check_mineral(self, u):
        if u.unit_type in sm.MINERAL_UNITS:
            return True
        else:
            return False

    def _check_vespene(self, u):
        if u.unit_type in sm.VESPENE_UNITS:
            return True
        else:
            return False

