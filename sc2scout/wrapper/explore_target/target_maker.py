from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.evade_enemy.evade_act_wrapper import EvadeActWrapper
from sc2scout.wrapper.explore_target.target_terminal_wrapper import TargetTerminalWrapper, \
TargetTerminalWrapperV1, TargetTerminalWrapperV2
from sc2scout.wrapper.explore_target.target_rwd_wrapper import ExploreTargetRwdWrapper, \
TargetSimpleRwdWrapper, TargetRoundTripRwdWrapper
from sc2scout.wrapper.explore_target.target_obs_wrapper import TargetObsWrapper, \
TargetObsWrapperV1, TargetObsWrapperV2, TargetObsWrapperV3, TargetObsWrapperV4, \
TargetObsWrapperV5, TargetObsWrapperV6
from sc2scout.wrapper.wrapper_factory import WrapperMaker

from baselines import deepq

class TargetMakerV1(WrapperMaker):
    def __init__(self):
        super(TargetMakerV1, self).__init__('target_v1')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapper(env, 128, 48, 200)
        env = ExploreTargetRwdWrapper(env, 128, 48, 200)
        env = TargetObsWrapper(env, 128, 48, 200)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256, 128])

class TargetMakerV2(WrapperMaker):
    def __init__(self):
        super(TargetMakerV2, self).__init__('target_v2')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV1(env)
        env = TargetSimpleRwdWrapper(env, 64, 24)
        env = TargetObsWrapperV1(env, 64, 256, 64, False)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class TargetMakerV3(WrapperMaker):
    def __init__(self):
        super(TargetMakerV3, self).__init__('target_v3')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV1(env)
        env = TargetSimpleRwdWrapper(env, 32, 12)
        env = TargetObsWrapperV1(env, 32, 128, 32, False)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class TargetMakerV4(WrapperMaker):
    def __init__(self):
        super(TargetMakerV4, self).__init__('target_v4')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV2(env, 32, 12, 400)
        env = TargetRoundTripRwdWrapper(env, 32, 12, 400)
        env = TargetObsWrapperV2(env, 32, 12, 400)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class TargetMakerV5(WrapperMaker):
    def __init__(self):
        super(TargetMakerV5, self).__init__('target_v5')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV1(env)
        env = TargetSimpleRwdWrapper(env, 32, 12)
        env = TargetObsWrapperV3(env, 32, 12, False)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class TargetMakerV6(WrapperMaker):
    def __init__(self):
        super(TargetMakerV6, self).__init__('target_v6')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV1(env)
        env = TargetSimpleRwdWrapper(env, 32, 12)
        env = TargetObsWrapperV4(env, 32, 12, 128, 32, False)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class TargetMakerV7(WrapperMaker):
    def __init__(self):
        super(TargetMakerV7, self).__init__('target_v7')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV1(env)
        env = TargetSimpleRwdWrapper(env, 32, 12)
        env = TargetObsWrapperV5(env, 32, 12)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class TargetMakerV8(WrapperMaker):
    def __init__(self):
        super(TargetMakerV8, self).__init__('target_v8')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        env = TargetTerminalWrapperV2(env, 32, 12, 500)
        env = TargetRoundTripRwdWrapper(env, 32, 12, 500)
        env = TargetObsWrapperV6(env, 32, 12, 500)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

