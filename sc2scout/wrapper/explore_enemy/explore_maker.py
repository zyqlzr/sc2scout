from sc2scout.wrapper.explore_enemy.action_wrapper import ZergScoutActWrapper
from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.explore_enemy.reward_wrapper import ZergScoutRwdWrapper, \
ZergScoutRwdWrapperV2, ZergScoutRwdWrapperV4, ZergScoutRwdWrapperV5
from sc2scout.wrapper.explore_enemy.auxiliary_wrapper import SkipFrame, \
TerminalWrapper, RoundTripTerminalWrapper
from sc2scout.wrapper.explore_enemy.observation_wrapper import ZergScoutObsWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

from baselines import deepq

class ExploreMakerV0(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV0, self).__init__('expore_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = ZergScoutRwdWrapper(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

class ExploreMakerV2(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV2, self).__init__('explore_v2')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env, skip_count=1)
        env = ZergScoutRwdWrapper(env)
        env = ZergScoutObsWrapper(env)
        env = TerminalWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

class ExploreMakerV6(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV6, self).__init__('expore_v6')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = ZergScoutRwdWrapperV2(env)
        env = ZergScoutObsWrapper(env)
        env = TerminalWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

class ExploreMakerV8(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV8, self).__init__('expore_v8')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = TerminalWrapper(env)
        env = ZergScoutRwdWrapperV4(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

class ExploreMakerV9(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV9, self).__init__('explore_v9')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = RoundTripTerminalWrapper(env)
        env = ZergScoutRwdWrapperV5(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([512, 256])

class ExploreMakerV10(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV10, self).__init__('expore_v10')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = TerminalWrapper(env)
        env = ZergScoutRwdWrapperV4(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([512, 256])

