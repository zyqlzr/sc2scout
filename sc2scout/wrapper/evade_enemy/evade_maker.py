from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.evade_enemy.evade_act_wrapper import EvadeActWrapper
from sc2scout.wrapper.evade_enemy.evade_rwd_wrapper import ScoutEvadeRwdWrapper, ScoutEvadeImgRwdWrapper
from sc2scout.wrapper.evade_enemy.evade_terminal_wrapper import EvadeTerminalWrapper, EvadeImgTerminalWrapper
from sc2scout.wrapper.evade_enemy.evade_obs_wrapper import ScoutEvadeObsWrapper, ScoutEvadeImgObsWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

from baselines import deepq

class EvadeMakerV0(WrapperMaker):
    def __init__(self):
        super(EvadeMakerV0, self).__init__('evade_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
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
        env = EvadeActWrapper(env)
        env = EvadeImgTerminalWrapper(env)
        env = ScoutEvadeImgRwdWrapper(env)
        env = ScoutEvadeImgObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], 
            hiddens=[256, 128])

