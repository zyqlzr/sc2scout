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
  NOOP = 8

scout_action_dict = {0: 'upper',
               1: 'left',
               2: 'down',
               3: 'right',
               4: 'upper-left',
               5: 'lower-left',
               6: 'lower-right',
               7: 'upper-right',
               8: 'noop'
              }

MOVE_RANGE = 1.0

class ZergScoutActWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ZergScoutActWrapper, self).__init__(env)
        self._reverse = False

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        print('action wrapper, reverse={}', self._reverse)
        return obs

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
        if self._reverse:
            old_action = action
            action = self.action_transfer(action)
            print('reverse, action{}->{}'.format(old_action, action))
        scout = self.env.unwrapped.scout()
        if action == ScoutMove.UPPER.value:
            pos = (scout.float_attr.pos_x, 
                   scout.float_attr.pos_y + MOVE_RANGE)
            #print('action upper,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.LEFT.value:
            pos = (scout.float_attr.pos_x + MOVE_RANGE,
                   scout.float_attr.pos_y)
            #print('action left,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.DOWN.value:
            pos = (scout.float_attr.pos_x,
                   scout.float_attr.pos_y - MOVE_RANGE)
            #print('action down,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.RIGHT.value:
            pos = (scout.float_attr.pos_x - MOVE_RANGE, 
                   scout.float_attr.pos_y)
            #print('action right,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.UPPER_LEFT.value:
            pos = (scout.float_attr.pos_x + MOVE_RANGE,
                   scout.float_attr.pos_y + MOVE_RANGE)
            #print('action upper_left,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.LOWER_LEFT.value:
            pos = (scout.float_attr.pos_x + MOVE_RANGE,
                   scout.float_attr.pos_y - MOVE_RANGE)
            #print('action lower_left,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.LOWER_RIGHT.value:
            pos = (scout.float_attr.pos_x - MOVE_RANGE,
                   scout.float_attr.pos_y - MOVE_RANGE)
            #print('action lower_right,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        elif action == ScoutMove.UPPER_RIGHT.value:
            pos = (scout.float_attr.pos_x - MOVE_RANGE,
                   scout.float_attr.pos_y + MOVE_RANGE)
            #print('action upper_right,scout:{} pos:{}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), pos))
        else:
            #print('action upper_right,scout:{} pos:None, action={}'.format(
            #       (scout.float_attr.pos_x, scout.float_attr.pos_y), action))
            pos = None
        return pos

    def _move_to_target(self, pos):
        scout = self.env.unwrapped.scout()
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = tp.ABILITY_ID.SMART.value
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        action.action_raw.unit_command.unit_tags.append(scout.tag)
        return action

    def _noop(self):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = tp.ABILITY_ID.INVALID.value
        return action

    def action_transfer(self, action):
        if action == ScoutMove.UPPER.value:
            return ScoutMove.DOWN.value
        elif action == ScoutMove.LEFT.value:
            return ScoutMove.RIGHT.value
        elif action == ScoutMove.DOWN.value:
            return ScoutMove.UPPER.value
        elif action == ScoutMove.RIGHT.value:
            return ScoutMove.LEFT.value
        elif action == ScoutMove.UPPER_LEFT.value:
            return ScoutMove.LOWER_RIGHT.value
        elif action == ScoutMove.LOWER_LEFT.value:
            return ScoutMove.UPPER_RIGHT.value
        elif action == ScoutMove.LOWER_RIGHT.value:
            return ScoutMove.UPPER_LEFT.value
        elif action == ScoutMove.UPPER_RIGHT.value:
            return ScoutMove.LOWER_LEFT.value
        else:
            pos = None
        return pos

    def judge_reverse(self):
        scout = self.env.unwrapped.scout()
        if scout.float_attr.pos_x < scout.float_attr.pos_y:
            return False
        else:
            return True

