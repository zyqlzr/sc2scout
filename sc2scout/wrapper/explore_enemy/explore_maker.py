from sc2scout.wrapper.explore_enemy.action_wrapper import ZergScoutActWrapper
from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.explore_enemy.reward_wrapper import ZergScoutRwdWrapper, \
ZergScoutRwdWrapperV1, ZergScoutRwdWrapperV2, ZergScoutRwdWrapperV3, \
ZergScoutRwdWrapperV4, ZergScoutRwdWrapperV5
from sc2scout.wrapper.explore_enemy.auxiliary_wrapper import SkipFrame, \
TerminalWrapper, CourseJudgeWrapper
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

class ExploreMakerV1(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV1, self).__init__('expore_v1')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = ZergScoutRwdWrapper(env)
        env = ZergScoutObsWrapper(env)
        env = TerminalWrapper(env)
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

class ExploreMakerV3(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV3, self).__init__('explore_v3')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = ZergScoutRwdWrapper(env)
        env = ZergScoutObsWrapper(env)
        env = TerminalWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 32])

class ExploreMakerV4(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV4, self).__init__('explore_v4')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env, skip_count=1)
        env = CourseJudgeWrapper(env)
        env = ZergScoutRwdWrapperV1(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 64, 32])

class ExploreMakerV5(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV5, self).__init__('explore_v5')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = CourseJudgeWrapper(env)
        env = ZergScoutRwdWrapperV1(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 64, 32])

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

class ExploreMakerV7(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV7, self).__init__('explore_v7')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = CourseJudgeWrapper(env)
        env = ZergScoutRwdWrapperV3(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 64, 32])

class ExploreMakerV8(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV8, self).__init__('expore_v8')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = ZergScoutRwdWrapperV4(env)
        env = ZergScoutObsWrapper(env)
        env = TerminalWrapper(env)
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
        env = CourseJudgeWrapper(env)
        env = ZergScoutRwdWrapperV5(env)
        env = ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.mlp([64, 64, 32])


