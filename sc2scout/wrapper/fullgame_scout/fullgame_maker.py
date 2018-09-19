from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_act_wrapper import FullGameActWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_rwd_wrapper import FullGameRwdWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_obs_wrapper import FullGameObsWrapper, \
FullGameObsMiniWrapper, FullGameObsMiniWrapperV1, FullGameObsMiniWrapperV2
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
        env = FullGameObsWrapper(env, 32, 12)
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
        env = FullGameObsMiniWrapper(env, 32, 12)
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
        env = FullGameObsMiniWrapperV1(env, 32, 12)
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
        env = FullGameObsMiniWrapperV2(env, 32, 12, 12)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        pass


