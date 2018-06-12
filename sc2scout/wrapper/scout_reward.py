from sc2scout.envs import scout_macro as sm

class Reward(object):
    def __init__(self, w):
        self.w = w
        self.rwd = 0

    def reset(self, obs, env):
        raise NotImplementedError

    def compute_rwd(self, obs, reward, env):
        raise NotImplementedError

class HomeReward(Reward):
    def __init__(self):
        super(HomeReward, self).__init__(1)
        self._last_dist = None

    def reset(self, obs, env):
        self._last_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, env):
        tmp_dist = self._compute_dist(env)
        if tmp_dist > self._last_dist:
            self.rwd = self.w * 1
        elif tmp_dist < self._last_dist:
            self.rwd = self.w * -1
        else:
            self.rwd = 0
        #print('home_rwd=', self.rwd)
        self._last_dist = tmp_dist

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     home[0], home[1])
        return dist

class EnemyBaseReward(Reward):
    def __init__(self):
        super(EnemyBaseReward, self).__init__(1)
        self._last_dist = None

    def reset(self, obs, env):
        self._last_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, env):
        tmp_dist = self._compute_dist(env)
        if tmp_dist < self._last_dist:
            self.rwd = self.w * 1
        elif tmp_dist > self._last_dist:
            self.rwd = self.w * -1
        else:
            self.rwd = 0
        self._last_dist = tmp_dist
        #print('enemy_rwd=', self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        enemy_base = env.enemy_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     enemy_base[0], enemy_base[1])
        return dist

class ViewEnemyReward(Reward):
    def __init__(self):
        super(ViewEnemyReward, self).__init__(3)
        self._unit_set = set([])

    def reset(self, obs, env):
        pass

    def compute_rwd(self, obs, reward, env):
        enemy_units = []
        units = obs.observation['units']
        for unit in units:
            if unit.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemy_units.append(unit.tag)

        count = 0
        for eu in enemy_units:
            if eu in self._unit_map:
                pass
            else:
                count += 1
                self._unit_map.add(eu)

        self.rwd = count * self.w
