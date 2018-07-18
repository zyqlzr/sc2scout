from pysc2.lib.typeenums import UNIT_TYPEID, ABILITY_ID
from s2clientprotocol import sc2api_pb2 as sc_pb

from sc2scout.agents.agent_base import AgentBase
import sc2scout.envs.scout_macro as sm
import random

class ZergBotAgent(AgentBase):
    def __init__(self):
        self._target = None
        self._range = 12
        self._defense_range = 7
        self._attack_range = 4

    def act(self, observation, reward, done):
        queen = None
        enemy = None
        for u in observation.observation['units']:
            if (u.int_attr.unit_type == UNIT_TYPEID.ZERG_QUEEN.value and 
                u.int_attr.alliance == sm.AllianceType.SELF.value):
                queen = u
            elif (u.int_attr.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value and
                  u.int_attr.alliance == sm.AllianceType.ENEMY.value):
                enemy = u

        if queen is None:
            print('ZergBotAgent cannot find queue')
            return [self._noop()]

        dist = sm.calculate_distance(queen.float_attr.pos_x, queen.float_attr.pos_y, 
                                     self._target[0], self._target[1])
        if self._check_defense(dist):
            #print('ZergBotAgent queen on defense')
            return [self._defense(queen)]
        if self._check_out_of_range(dist):
            #print('ZergBotAgent queen defense while out of range')
            return [self._defense(queen)]

        if enemy is None:
            #print('ZergBotAgent queen random')
            return [self._random(queen)]

        diff_dist = sm.calculate_distance(queen.float_attr.pos_x, 
                                          queen.float_attr.pos_y,
                                          enemy.float_attr.pos_x,
                                          enemy.float_attr.pos_y)
        if diff_dist < self._attack_range:
            #print('ZergBotAgent queen attack')
            return [self._attack_target(queen, [enemy.float_attr.pos_x,
                                        enemy.float_attr.pos_y])]
        else:
            #print('ZergBotAgent queen catch-up')
            return [self._move_to_target(queen, [enemy.float_attr.pos_x,
                                        enemy.float_attr.pos_y])]

    def reset(self, env):
        self._target = env.unwrapped.enemy_base()
        self._out_of_range = False
        self._on_defense = False
        radius = self._range / 2
        self._x_low = self._target[0] - radius
        self._x_high = self._target[0] + radius
        self._y_low = self._target[1] - radius
        self._y_high = self._target[1] + radius
        print('ZergBotAgent Reset, target=', self._target)

    def _move_to_target(self, u, pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = ABILITY_ID.SMART.value
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        action.action_raw.unit_command.unit_tags.append(u.tag)
        return action

    def _attack_target(self, u, pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = ABILITY_ID.ATTACK_ATTACK.value
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        action.action_raw.unit_command.unit_tags.append(u.tag)
        return action

    def _noop(self):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = ABILITY_ID.INVALID.value
        return action

    def _defense(self, u):
        if not self._on_defense:
            self._on_defense = True
        return self._move_to_target(u, self._target)

    def _random(self, u):
        pos_x = random.uniform(self._x_low, self._x_high)
        pos_y = random.uniform(self._y_low, self._y_high)
        return self._move_to_target(u, (pos_x, pos_y))

    def _check_defense(self, dist):
        if self._on_defense:
            if dist < self._defense_range:
                self._on_defense = False
        return self._on_defense

    def _check_out_of_range(self, dist):
        if dist > self._range:
            if not self._out_of_range:
                self._out_of_range = True

            return True
        return False
