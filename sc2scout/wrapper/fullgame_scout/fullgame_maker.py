from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_act_wrapper import FullGameActWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_rwd_wrapper import FullGameRwdWrapper
from sc2scout.wrapper.fullgame_scout.fullgame_obs_wrapper import FullGameObsWrapper
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

