import gym
import numpy as np
from enum import Enum, unique

from s2clientprotocol import sc2api_pb2 as sc_pb
import pysc2.lib.typeenums as tp
from pysc2.lib.typeenums import UNIT_TYPEID

@unique
class ScoutMove(Enum):
  UPPER = 0
  LEFT = 1
  DOWN = 2
  RIGHT = 3
  UPPER_LEFT = 4
  LOWER_LEFT = 5
  LOWER_RIGHT = 6
  UPPER_RIGHT = 7

scout_action_dict = {0: 'upper',
               1: 'left',
               2: 'down',
               3: 'right',
               4: 'upper-left',
               5: 'lower-left',
               6: 'lower-right',
               7: 'upper-right'
              }

MOVE_RANGE = 1.0

class ZergScoutActWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ZergScoutActWrapper, self).__init__(env)
        self._init_action_space()
        self._scout = None

    def _reset(self):
        print('*****action wrapper reset*****')
        obs = self.env._reset()
        self._select_scout_from_customized(obs)
        print('scout_unit=', self._scout.tag)
        return obs

    def _init_action_space(self):
        self.action_space = gym.spaces.Discrete(8)

    def _step(self, action):
        action = self.action(action)
        return self.env._step(action)

    def _action(self, action):
        pos = self._calcuate_pos_by_action(action)
        if pos is None:
            return [[self._noop()]]
        else:
            return [[self._move_to_target(pos)]]

    def _reverse_action(self, action):
        raise NotImplementedError()

    def _calcuate_pos_by_action(self, action):
        if action == ScoutMove.UPPER.value:
            pos = (self._scout.float_attr.pos_x, 
                   self._scout.float_attr.pos_y + MOVE_RANGE)
        elif action == ScoutMove.LEFT.value:
            pos = (self._scout.float_attr.pos_x + MOVE_RANGE,
                   self._scout.float_attr.pos_y)
        elif action == ScoutMove.DOWN.value:
            pos = (self._scout.float_attr.pos_x,
                   self._scout.float_attr.pos_y - MOVE_RANGE)
        elif action == ScoutMove.RIGHT.value:
            pos = (self._scout.float_attr.pos_x - MOVE_RANGE, 
                   self._scout.float_attr.pos_y)
        elif action == ScoutMove.UPPER_LEFT.value:
            pos = (self._scout.float_attr.pos_x + MOVE_RANGE,
                   self._scout.float_attr.pos_y + MOVE_RANGE)
        elif action == ScoutMove.LOWER_LEFT.value:
            pos = (self._scout.float_attr.pos_x + MOVE_RANGE,
                   self._scout.float_attr.pos_y - MOVE_RANGE)
        elif action == ScoutMove.LOWER_RIGHT.value:
            pos = (self._scout.float_attr.pos_x - MOVE_RANGE,
                   self._scout.float_attr.pos_y - MOVE_RANGE)
        elif action == ScoutMove.UPPER_RIGHT.value:
            pos = (self._scout.float_attr.pos_x - MOVE_RANGE,
                   self._scout.float_attr.pos_y + MOVE_RANGE)
        else:
            pos = None
        return pos

    def _move_to_target(self, pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = tp.ABILITY_ID.SMART.value
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        action.action_raw.unit_command.unit_tags.append(self._scout.tag)
        return action

    def _noop(self):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = tp.ABILITY_ID.INVALID.value
        return action

    def _select_scout_from_customized(self, obs):
        units = obs.observation['units']
        # update scout
        if self._scout is None:
            for u in units:
                if u.int_attr.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value:
                    self._scout = u

