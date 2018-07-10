import gym
from sc2scout.envs.sc2_gym_env import SC2GymEnv
from sc2scout.envs import scout_macro as sm
from pysc2.lib.typeenums import UNIT_TYPEID
import numpy as np


MAP_SIZE = {
    'Simple64': (88, 96),
    'ScoutSimple64': (88, 96),
    'ScoutSimple64WithQueue':(88,96),
    'ScoutSimple64WithQueue_evade':(88,96),
    'scout_evade':(32,32),
    'scout_evade_multi':(32,32),
    'AbyssalReef': (200, 176),
    'ScoutAbyssalReef': (200, 176),
    'Acolyte': (168, 200),
}

class ZergEvadeEnv(SC2GymEnv):
    def __init__(self, **kwargs):
        super(ZergEvadeEnv, self).__init__(**kwargs)
        self._init_action_space()
        self._scout = None
        self._enemy = None
        self._view_enemy = False
        self._map_size = None
        self._init_map_size(kwargs['map_name'])

    def _init_action_space(self):
        self.action_space = gym.spaces.Discrete(8)

    def _init_map_size(self, map_name):
        self._map_size = MAP_SIZE[map_name]
        print('map_name={}, MapSize={}'.format(map_name, self._map_size))

    def _reset(self):
        self._scout = None
        self._enemy = None
        self._view_enemy = False

        obs = super(ZergEvadeEnv, self)._reset()
        self._init_scout_and_enemy(obs)
        print('*** ZergEvadeEnv scout_unit=', self._scout.tag)
        return obs

    def _step(self, action):
        obs, rwd, done, other = super(ZergEvadeEnv, self)._step(action)
        self._update_scout(obs)
        self._update_enemy(obs)
        return obs, rwd, done, other

    def scout(self):
        return self._scout

    def view_enemy(self):
        return self._view_enemy

    def enemy(self):
        return self._enemy

    def map_size(self):
        return self._map_size


    def _update_scout(self, obs):
        units = obs.observation['units']
        for u in units:
            if u.tag == self._scout.tag:
                self._scout = u
                #print('update scout,pos=', (self._scout.float_attr.pos_x,
                #      self._scout.float_attr.pos_y))
        #print('update scout, {},{}'.format(self._scout.tag,
        #        (self._scout.float_attr.pos_x, self._scout.float_attr.pos_y)))

    def _update_enemy(self, obs):
        self._view_enemy = False
        self._enemy = None
        units = obs.observation['units']
        for u in units:
            if self._check_enemy(u):
                if not self._view_enemy:
                    self._view_enemy = True

                if self._enemy is None:
                    self._enemy = u
                else:
                    temp_dist = self._unit_dist(self._scout, u)
                    curr_dist = self._unit_dist(self._scout,self._enemy)
                    if temp_dist<curr_dist: # only consider the most closed enemy
                        self._enemy = u


    def _init_scout_and_enemy(self,obs):
        units = obs.observation['units']
        for u in units:
            if u.int_attr.unit_type == UNIT_TYPEID.ZERG_ZERGLING.value:
                if self._scout is None:
                    self._scout = u
                    print('find scout,unit=', self._scout.tag, ',unit_pos=',
                          (self._scout.float_attr.pos_x, self._scout.float_attr.pos_y))

        for u in units:
            if self._check_enemy(u):
                if not self._view_enemy:
                    self._view_enemy = True

                if self._enemy is None:
                    self._enemy = u
                else:
                    temp_dist = self._unit_dist(self._scout, u)
                    curr_dist = self._unit_dist(self._scout,self._enemy)
                    if temp_dist<curr_dist: # only consider the most closed enemy
                        self._enemy = u

        if self._view_enemy:
            print('find enemy,unit=', self._enemy.tag, ',unit_pos=',
                          (self._enemy.float_attr.pos_x, self._enemy.float_attr.pos_y))


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

    def _check_enemy(self, u):
        if u.unit_type in sm.COMBAT_UNITS and \
                        u.int_attr.alliance == sm.AllianceType.ENEMY.value:
            return True
        else:
            return False