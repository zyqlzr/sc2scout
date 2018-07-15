from sc2scout.wrapper.explore_enemy.action_wrapper import ZergScoutActWrapper
from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.evade_enemy.evade_rwd_wrapper import ScoutEvadeRwdWrapper
from sc2scout.wrapper.evade_enemy.evade_terminal_wrapper import EvadeTerminalWrapper
from sc2scout.wrapper.evade_enemy.evade_obs_wrapper import ScoutEvadeObsWrapper, ScoutEvadeBlendObsWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

from baselines import deepq

class EvadeMakerV0(WrapperMaker):
    def __init__(self):
        super(EvadeMakerV0, self).__init__('evade_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = EvadeTerminalWrapper(env)
        env = ScoutEvadeRwdWrapper(env)
        env = ScoutEvadeObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

class EvadeMakerV1(WrapperMaker):
    def __init__(self):
        super(EvadeMakerV1, self).__init__('evade_v1')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = ScoutEvadeBlendObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return None

