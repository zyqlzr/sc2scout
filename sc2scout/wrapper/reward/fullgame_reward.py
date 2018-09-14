from sc2scout.envs import scout_macro as sm
from sc2scout.wrapper.reward.reward import Reward

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

