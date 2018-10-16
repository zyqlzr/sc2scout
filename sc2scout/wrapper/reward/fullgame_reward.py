from sc2scout.envs import scout_macro as sm
from sc2scout.wrapper.reward.reward import Reward
import math

MAX_VIEW_RANGE = 12

class FullGameViewRwd(Reward):
    def __init__(self, weight=3):
        super(FullGameViewRwd, self).__init__(weight)
        self._unit_set = set([])
        self._building_type_set = set([])

    def reset(self, obs, env):
        self._unit_set = set([])
        self._building_type_set = set([])

    def compute_rwd(self, obs, reward, done, env):
        enemy_units = []
        enemy_building_types = []
        find_base = False
        units = obs.observation['units']
        for unit in units:
            if unit.int_attr.alliance == sm.AllianceType.ENEMY.value:
                if not self._check_view(env, unit):
                    continue

                if unit.unit_type in sm.BUILDING_UNITS:
                    enemy_building_types.append(unit.unit_type)
                else:
                    enemy_units.append(unit.tag)

        ucount = 0
        for eu in enemy_units:
            if eu in self._unit_set:
                pass
            else:
                ucount += 1
                self._unit_set.add(eu)

        bcount = 0
        for eb in enemy_building_types:
            if eb in self._building_type_set:
                pass
            else:
                bcount += 1
                self._building_type_set.add(eb)

        self.rwd = ucount * self.w + bcount * self.w * 4
        #print('ucount={},bcount={},len_unit={},len_bt={}'.format(
        #      ucount, bcount, len(self._unit_set), len(self._building_type_set)))


    def _check_view(self, env, unit):
        scout = env.scout()
        dist = sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     unit.float_attr.pos_x,
                                     unit.float_attr.pos_y)
        if dist <= MAX_VIEW_RANGE:
            return True
        else:
            return False


class FullGameInTargetRangeRwd(Reward):
    def __init__(self, range_width, weight=1):
        super(FullGameInTargetRangeRwd, self).__init__(weight)
        self._target = None
        self._range_width = range_width
 
    def reset(self, obs, env):
        self._target = env.unwrapped.enemy_base()

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        in_range = self._in_range(env)
        if in_range:
            self.rwd = 0
        else:
            self.rwd = -1 * self.w
        #print('InRange scout=({}, {}), Rwd={}'.format(
        #    scout.float_attr.pos_x, scout.float_attr.pos_y, self.rwd))

    def _in_range(self, env):
        scout = env.unwrapped.scout()
        dist = sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     self._target[0],
                                     self._target[1])
        if dist <= self._range_width:
            return True
        else:
            return False


class FullGameMoveToTarget(Reward):
    def __init__(self, target_range, weight=1):
        super(FullGameMoveToTarget, self).__init__(weight)
        self._target_range = target_range

    def reset(self, obs, env):
        self._curr_dist = self._compute_curr_dist(env)
        self._max_dist = self._compute_curr_dist(env)
        self._rwd_sum = 0.0
        self._max_rwd_sum = 100.0
        self._last_left_rwd = 100

    def compute_rwd(self, obs, reward, done, env):
        curr_dist = self._compute_curr_dist(env)
        survive = env.unwrapped.scout_survive()
        if curr_dist == self._curr_dist:
            self.rwd = 0
            return

        rwd = self._compute_rwd(curr_dist)
        self.rwd = self.w * rwd
        self._curr_dist = curr_dist
        #print('underattack rwd; rwd={}, rwd_sum={}'.format(self.rwd, self._rwd_sum))

    def _compute_rwd(self, curr_dist):
        left_rwd = math.floor(int((curr_dist/ self._max_dist) * self._max_rwd_sum))
        rwd = self._last_left_rwd - left_rwd
        self._last_left_rwd = left_rwd
        self._rwd_sum += rwd
        return rwd

    def _compute_curr_dist(self, env):
        scout = env.unwrapped.scout()
        enemy_pos = env.unwrapped.enemy_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     enemy_pos[0], enemy_pos[1])
        return dist - self._target_range


