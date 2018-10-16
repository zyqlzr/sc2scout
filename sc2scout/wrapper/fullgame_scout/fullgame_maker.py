from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_act_wrapper import FullGameActWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_rwd_wrapper import FullGameRwdWrapper, \
FullGameRwdWrapperV1, FullGameRwdWrapperV2
from sc2scout.wrapper.fullgame_scout.fullgame_obs_wrapper import FullGameObsWrapper, \
FullGameObsMiniWrapper, FullGameObsMiniWrapperV1, FullGameObsMiniWrapperV2, \
FullGameObsWrapperV1, FullGameObsWrapperV2
from sc2scout.wrapper.fullgame_scout.fullgame_terminal_wrapper import FullGameTerminalWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

class FullGameMaker(WrapperMaker):
    def __init__(self):
        super(FullGameMaker, self).__init__('fullgame_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapper(env, 32, 12)
        env = FullGameObsWrapper(env, 32)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


class FullGameMiniMaker(WrapperMaker):
    def __init__(self):
        super(FullGameMiniMaker, self).__init__('fullgame_mini_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapper(env, 32, 12)
        env = FullGameObsMiniWrapper(env, 32)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


class FullGameMiniMakerV1(WrapperMaker):
    def __init__(self):
        super(FullGameMiniMakerV1, self).__init__('fullgame_mini_v1')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapper(env, 32, 12)
        env = FullGameObsMiniWrapperV1(env, 32)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


class FullGameMiniMakerV2(WrapperMaker):
    def __init__(self):
        super(FullGameMiniMakerV2, self).__init__('fullgame_mini_v2')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapper(env, 32, 12)
        env = FullGameObsMiniWrapperV2(env, 32, 12)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class FullGameMiniMakerV3(WrapperMaker):
    def __init__(self):
        super(FullGameMiniMakerV3, self).__init__('fullgame_mini_v3')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapper(env, 32, 12)
        env = FullGameObsWrapperV1(env, 32, 12)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


class FullGameMakerV1(WrapperMaker):
    def __init__(self):
        super(FullGameMakerV1, self).__init__('fullgame_v1')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapperV1(env, 20)
        env = FullGameObsMiniWrapperV1(env, 32)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


class FullGameMakerV2(WrapperMaker):
    def __init__(self):
        super(FullGameMakerV2, self).__init__('fullgame_v2')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapperV1(env, 40)
        env = FullGameObsWrapperV1(env, 32, 12)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass

class FullGameMakerV3(WrapperMaker):
    def __init__(self):
        super(FullGameMakerV3, self).__init__('fullgame_v3')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = FullGameActWrapper(env)
        env = FullGameTerminalWrapper(env)
        env = FullGameRwdWrapperV2(env, 40)
        env = FullGameObsWrapperV2(env, 32, 12, 40)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


