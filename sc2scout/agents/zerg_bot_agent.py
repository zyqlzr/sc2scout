from pysc2.lib.typeenums import UNIT_TYPEID, ABILITY_ID
from s2clientprotocol import sc2api_pb2 as sc_pb

from sc2scout.agents.agent_base import AgentBase

class ZergBotAgent(AgentBase):
    def __init__(self):
        pass

    def act(self, observation, reward, done):
        zergling = None
        enemy = None
        for u in observation.observation['units']:
            if u.int_attr.unit_type == UNIT_TYPEID.ZERG_ZERGLING.value:
                zergling = u
            elif u.int_attr.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value:
                enemy = u

        if zergling is None or enemy is None:
            return [self._noop()]

        return [self._move_to_target(zergling, [enemy.float_attr.pos_x,
                                            enemy.float_attr.pos_y])]

    def reset(self):
        pass

    def _move_to_target(self, u, pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = ABILITY_ID.SMART.value
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        action.action_raw.unit_command.unit_tags.append(u.tag)
        return action

    def _noop(self):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = ABILITY_ID.INVALID.value
        return action

