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


class FullGameQuickArrived(Reward):
    def __init__(self, target_range, weight=1):
        super(FullGameQuickArrived, self).__init__(weight)
        self._target_range = target_range

    def reset(self, obs, env):
        pass

    def compute_rwd(self, obs, reward, done, env):
        pass


class FullGameViewRwdV1(Reward):
    def __init__(self, weight=1):
        super(FullGameViewRwdV1, self).__init__(weight)
        self._unit_set = set([])
        self._force_set = set([])
        self._building_type_set = set([])
        self._base_id_set = set([])

    def reset(self, obs, env):
        self._unit_set = set([])
        self._force_set = set([])
        self._building_type_set = set([])
        self._base_id_set = set([])

    def compute_rwd(self, obs, reward, done, env):
        force_units = []
        other_units = []
        building_types = []
        base_ids = []

        enemys = self._unit_dispatch(obs)
        for u in enemys:
            if not self._check_view(env, u):
                continue

            if u.unit_type in sm.BASE_UNITS:
                bid = env.unwrapped.get_id_by_pos((u.float_attr.pos_x, 
                                                   u.float_attr.pos_y))
                base_ids.append(bid)
            elif u.unit_type in sm.BUILDING_UNITS:
                building_types.append(u.unit_type)
            elif u.unit_type in sm.COMBAT_ATTACK_UNITS:
                force_units.append(u.tag)
            else:
                other_units.append(u.tag)

        #print('FullGameViewRwdV1 force_units={},other_units={},building_types={},base_ids={}',
        #      len(force_units), len(other_units), len(building_types), len(base_ids))
        base_count = 0
        bt_count = 0
        force_count = 0
        other_count = 0
        for bid in base_ids:
            if bid in self._base_id_set:
                pass
            else:
                base_count += 1
                self._base_id_set.add(bid)

        for bt in building_types:
            if bt in self._building_type_set:
                pass
            else:
                bt_count += 1
                self._building_type_set.add(bt)

        for fu in force_units:
            if fu in self._force_set:
                pass
            else:
                force_count += 1
                self._force_set.add(fu)

        for ou in other_units:
            if ou in self._unit_set:
                pass
            else:
                other_count += 1
                self._unit_set.add(ou)

        self.rwd = 0
        self.rwd += other_count *  self.w 
        self.rwd += force_count * self.w * 2
        self.rwd += bt_count * self.w * 10
        self.rwd += base_count * self.w * 50
        #print('FullGameViewRwdV1 base={},bt={},force={},other={}'.format(
        #      len(self._base_id_set), len(self._building_type_set), 
        #      len(self._force_set), len(self._unit_set)))
        #print('FullGameViewRwdV1 view_rwd=', self.rwd)

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

    def _unit_dispatch(self, obs):
        units = obs.observation['units']
        enemys = []
        for u in units:
            if u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        return enemys


class FullGameFinalRwd(Reward):
    def __init__(self, weight=50):
        super(FullGameFinalRwd, self).__init__(weight)
        self._max_step = 2000.
        self._threshold_step = 1000.

    def reset(self, obs, env):
        pass

    def compute_rwd(self, obs, reward, done, env):
        if done:
            survive = env.unwrapped.scout_survive()
            if survive:
                self.rwd = 1 * self.w
            else:
                step_num = env.unwrapped.step_number()
                if step_num > self._max_step:
                    self.rwd = 1 * self.w
                elif step_num >= self._threshold_step:
                    self.rwd = step_num / self._max_step * self.w
                else:
                    self.rwd = -1 * (self._threshold_step - step_num) / self._threshold_step * self.w
            print('final_rwd=', self.rwd)
        else:
            self.rwd = 0


