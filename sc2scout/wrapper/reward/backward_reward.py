from sc2scout.wrapper.reward.reward import Reward
from sc2scout.wrapper.util.dest_range import DestRange

class BackwardRwd(Reward):
    def __init__(self, weight=1):
        super(BackwardRwd, self).__init__(weight)
        self._dest = None

    def reset(self, obs, env):
        home = env.unwrapped.owner_base()
        self._dest = DestRange(home, dest_range=16)

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        if self._dest.in_range((scout.float_attr.pos_x, scout.float_attr.pos_y)):
            self.rwd = 0
        else:
            self.rwd = -1 * self.w

class BackwardFinalRwd(Reward):
    def __init__(self, weight=50):
        super(BackwardFinalRwd, self).__init__(weight)
        self._dest = None

    def reset(self, obs, env):
        home = env.unwrapped.owner_base()
        self._dest = DestRange(home, dest_range=16)

    def compute_rwd(self, obs, reward, done, env):
        if done:
            scout = env.unwrapped.scout()
            if self._dest.in_range((scout.float_attr.pos_x, scout.float_attr.pos_y)):
                self.rwd = 1 * self.w
            else:
                self.rwd = 0 
            #print('final_rwd=', self.rwd)
        else:
            self.rwd = 0

