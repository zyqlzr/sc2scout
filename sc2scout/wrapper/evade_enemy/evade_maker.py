from sc2scout.wrapper.evade_enemy.action_wrapper import ZergScoutActWrapper
from sc2scout.wrapper.evade_enemy.zerg_evade_wrapper import ZergEvadeWrapper
from sc2scout.wrapper.evade_enemy.reward_wrapper import ZergScoutRwdWrapper
from sc2scout.wrapper.evade_enemy.auxiliary_wrapper import SkipFrame,TerminalWrapper
from sc2scout.wrapper.evade_enemy.observation_wrapper import ZergScoutObsWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

from baselines import deepq

class EvadeMakerV0(WrapperMaker):
    def __init__(self):
        super(EvadeMakerV0, self).__init__('evade_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        # env = SkipFrame(env)
        env = TerminalWrapper(env)
        env = ZergScoutRwdWrapper(env)
        env = ZergScoutObsWrapper(env)
        env = ZergEvadeWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

