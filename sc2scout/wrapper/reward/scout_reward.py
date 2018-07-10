from sc2scout.envs import scout_macro as sm
from sc2scout.wrapper.reward.reward import Reward
from sc2scout.wrapper.util.dest_range import DestRange
import math

ARRIVED_DIST_GAP = 4

class HomeReward(Reward):
    def __init__(self, back=False, negative=False):
        super(HomeReward, self).__init__(1)
        self._last_dist = None
        self._back = back
        self._negative = negative

    def reset(self, obs, env):
        self._last_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        tmp_dist = self._compute_dist(env)
        if self._back:
            if self._negative:
                if tmp_dist > self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
                if tmp_dist > self._last_dist:
                    self.rwd = self.w * -1
                elif tmp_dist < self._last_dist:
                    self.rwd = self.w * 1
                else:
                    self.rwd = 0
        else:
            if self._negative:
                if tmp_dist < self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
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

class HomeArrivedReward(Reward):
    def __init__(self, weight=10):
        super(HomeArrivedReward, self).__init__(weight)
        self._once = False

    def reset(self, obs, env):
        self._once = False

    def compute_rwd(self, obs, reward, done, env):
        if self._once:
            self.rwd = 0
            return
        tmp_dist = self._compute_dist(env)
        if tmp_dist <= ARRIVED_DIST_GAP:
            self.rwd = 1 * self.w
            self._once = True
        else:
            self.rwd = 0
        print('home_arrived reward=', self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     home[0], home[1])
        return dist

class EnemyBaseReward(Reward):
    def __init__(self, back=False, negative=False):
        super(EnemyBaseReward, self).__init__(1)
        self._last_dist = None
        self._back = back
        self._negative = negative

    def reset(self, obs, env):
        self._last_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        tmp_dist = self._compute_dist(env)
        if self._back:
            if self._negative:
                if tmp_dist < self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
                if tmp_dist < self._last_dist:
                    self.rwd = self.w * -1
                elif tmp_dist > self._last_dist:
                    self.rwd = self.w * 1
                else:
                    self.rwd = 0
        else:
            if self._negative:
                if tmp_dist > self._last_dist:
                    self.rwd = self.w * -1
                else:
                    self.rwd = 0
            else:
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

class EnemyBaseArrivedReward(Reward):
    def __init__(self, weight=10):
        super(EnemyBaseArrivedReward, self).__init__(weight)
        self._once = False

    def reset(self, obs, env):
        self._once = False

    def compute_rwd(self, obs, reward, done, env):
        if self._once:
            self.rwd = 0
            return
        tmp_dist = self._compute_dist(env)
        if tmp_dist <= ARRIVED_DIST_GAP:
            self.rwd = 1 * self.w
            self._once = True
        else:
            self.rwd = 0
        #print('enemy_base_arrived reward=', self.rwd)

    def _compute_dist(self, env):
        scout = env.scout()
        enemy_base = env.enemy_base()
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     enemy_base[0], enemy_base[1])
        return dist

class ViewEnemyReward(Reward):
    def __init__(self, weight=3):
        super(ViewEnemyReward, self).__init__(weight)
        self._unit_set = set([])
        ''' the tag of enemybase will be repeatly changed while scout is nearby'''
        self._enemy_base_once = False

    def reset(self, obs, env):
        self._unit_set = set([])
        self._enemy_base_once = False

    def compute_rwd(self, obs, reward, done, env):
        enemy_units = []
        find_base = False
        units = obs.observation['units']
        for unit in units:
            if unit.int_attr.alliance == sm.AllianceType.ENEMY.value:
                if unit.unit_type in sm.BASE_UNITS:
                    if not self._enemy_base_once:
                        self._enemy_base_once = True
                        find_base = True
                else:
                    enemy_units.append(unit.tag)

        count = 0
        for eu in enemy_units:
            if eu in self._unit_set:
                pass
            else:
                count += 1
                self._unit_set.add(eu)
        if find_base:
            count += 1

        self.rwd = count * self.w
        #print('view enemy count=', count, ';reward=', self.rwd)

MIN_DIST_ERROR = 2.0

class MinDistReward(Reward):
    def __init__(self, negative=False):
        super(MinDistReward, self).__init__(1)
        self._min_dist = None
        self._negative = negative

    def reset(self, obs, env):
        self._min_dist = self._compute_dist(env)

    def compute_rwd(self, obs, reward, done, env):
        tmp_dist = self._compute_dist(env)
        if self._negative:
            if tmp_dist > self._min_dist + MIN_DIST_ERROR:
                self.rwd = self.w * -1
            else:
                self.rwd = 0
        else:
            if tmp_dist > self._min_dist + MIN_DIST_ERROR:
                self.rwd = self.w * -1
            else:
                self.rwd = self.w * 1

        if self._min_dist > tmp_dist:
            self._min_dist = tmp_dist

    def _compute_dist(self, env):
        scout = env.scout()
        home = env.owner_base()
        enemy_base = env.enemy_base()
        home_dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     home[0], home[1])

        enemy_dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     enemy_base[0], enemy_base[1])
        return (home_dist + enemy_dist)

class OnewayFinalReward(Reward):
    def __init__(self, weight=50):
        super(OnewayFinalReward, self).__init__(weight)

    def reset(self, obs, env):
        self._dest = DestRange(env.enemy_base())

    def compute_rwd(self, obs, reward, done, env):
        self._compute_rwd(env)
        if done:
            if self._dest.hit:
                #print('compute final rwd, hit rwd=', self.w * 2)
                self.rwd = self.w * 2
            elif self._dest.enter:
                #print('compute final rwd, enter rwd=', self.w * 1)
                self.rwd = self.w * 1
            else:
                self.rwd = self.w * -1
        else:
            self.rwd = 0

    def _compute_rwd(self, env):
        scout = env.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)

        self._dest.check_enter(pos)
        self._dest.check_hit(pos)
        self._dest.check_leave(pos)

class RoundTripFinalReward(Reward):
    def __init__(self, weight=50):
        super(RoundTripFinalReward, self).__init__(weight)
        self._back = False

    def reset(self, obs, env):
        self._dest = DestRange(env.enemy_base())
        self._src = DestRange(env.owner_base())

    def compute_rwd(self, obs, reward, done, env):
        scout = env.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        self._check(pos)
        if done:
            if self._dest.hit and self._src.hit:
                self.rwd = self.w * 2
            elif self._dest.enter and self._src.enter:
                self.rwd = self.w * 1
            else:
                self.rwd = self.w * -1
        else:
            self.rwd

    def _check(self, pos):
        if not self._back:
            self._check_dest(pos)
        else:
            self._check_src(pos)
        if self._dest.enter and self._dest.leave:
            if not self._back:
                self._back = True

    def _check_dest(self, pos):
        self._dest.check_enter(pos)
        self._dest.check_hit(pos)
        self._dest.check_leave(pos)

    def _check_src(self, pos):
        self._src.check_enter(pos)
        self._src.check_hit(pos)
        self._src.check_leave(pos)

class EvadeTimeReward(Reward):
    def __init__(self, weight=1):
        super(EvadeTimeReward, self).__init__(weight)

    def reset(self, obs, env):
        self._curr_step = 0

    def compute_rwd(self, obs, reward, done, env):
        curr_step = env.curr_step()
        game_step_per_episode = env.episode_length()
        normorlized_curr_step = float(curr_step)/game_step_per_episode

        self.rwd = 1 - math.pow((1-normorlized_curr_step),0.4)
        self.rwd = self.w * self.rwd
        print('EvadeTimeReward reward=', self.rwd)

class EvadeSpaceReward(Reward):
    def __init__(self, weight=1):
        super(EvadeSpaceReward, self).__init__(weight)


    def reset(self, obs, env):
        self._map_size = env.map_size()

    def compute_rwd(self, obs, reward, done, env):
        viewEnemy = env.view_enemy()

        enemy = env.enemy()
        if not viewEnemy:
            self.rwd = 0
        else:
            if enemy:
                scout_x = env.scout().float_attr.pos_x/float(self._map_size[0])
                scout_y = env.scout().float_attr.pos_y/float(self._map_size[1])
                enemy_x = env.enemy().float_attr.pos_x/float(self._map_size[0])
                enemy_y = env.enemy().float_attr.pos_y/float(self._map_size[1])
                curr_dist = env._calculate_distances(scout_x,scout_y,enemy_x,enemy_y)

                self.rwd = math.pow(curr_dist,0.4) - 1
        self.rwd = self.dynamicWeight(env) * self.rwd

        print('EvadeSpaceReward reward=', self.rwd)

    def dynamicWeight(self,env):
        curr_step = env.curr_step()
        game_step_per_episode = env.episode_length()
        return self.w * math.pow(float(curr_step)/game_step_per_episode,0.4)*10000

class EvadeHealthReward(Reward):
    def __init__(self, weight=1):
        super(EvadeHealthReward, self).__init__(weight)

    def reset(self, obs, env):
        self.scout_max_health = env.scout().float_attr.health_max

    def compute_rwd(self, obs, reward, done, env):
        normorlized_health = env.scout().float_attr.health/float(self.scout_max_health)
        self.rwd = math.pow(normorlized_health, 0.4) - 1
        self.rwd = self.dynamicWeight(env) * self.rwd

        print('EvadeHealthReward reward=', self.rwd)


    def dynamicWeight(self,env):
        curr_step = env.curr_step()
        game_step_per_episode = env.episode_length()
        return self.w * math.pow(float(curr_step)/game_step_per_episode,0.4)*10000


