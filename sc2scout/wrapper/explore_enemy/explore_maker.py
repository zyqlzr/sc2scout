from sc2scout.wrapper.explore_enemy.action_wrapper import ZergScoutActWrapper
from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.explore_enemy.reward_wrapper import ZergScoutRwdWrapper
from sc2scout.wrapper.explore_enemy.auxiliary_wrapper import SkipFrame
from sc2scout.wrapper.explore_enemy.observation_wrapper import ZergScoutObsWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

class ExploreMakerV0(WrapperMaker):
    def __init__(self):
        super(ExploreMakerV0, self).__init__('expore_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZergScoutActWrapper(env)
        env = SkipFrame(env)
        env = ZergScoutRwdWrapper(env)
        env =ZergScoutObsWrapper(env)
        env = ZergScoutWrapper(env)
        return env
