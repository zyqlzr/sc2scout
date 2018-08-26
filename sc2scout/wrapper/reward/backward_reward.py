from sc2scout.envs import scout_macro as sm
from sc2scout.wrapper.reward.reward import Reward
from sc2scout.wrapper.util.dest_range import DestRange

class BackwardRwd(Reward):
    def __init__(self, weight=1):
        super(BackwardRwd, self).__init__(weight)
        self._dest = None
        self._last_dist = None
        self._home = None

    def reset(self, obs, env):
        scout = env.unwrapped.scout()
        self._home = env.unwrapped.owner_base()
        self._dest = DestRange(self._home, dest_range=16)
        self._last_dist = None

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        if self._last_dist is None:
            self._last_dist = self._compute_dist(scout)

        if not self._dest.in_range((scout.float_attr.pos_x, 
                                    scout.float_attr.pos_y)):
            tmp_dist = self._compute_dist(scout)
            if tmp_dist > self._last_dist:
                self.rwd = self.w * -1
            else:
                self.rwd = 0
            self._last_dist = tmp_dist
        else:
            self.rwd = 0
        print('backreward=', self.rwd)

    def _compute_dist(self, scout):
        dist = sm.calculate_distance(scout.float_attr.pos_x, 
                                     scout.float_attr.pos_y, 
                                     self._home[0], self._home[1])
        return dist

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
            print('backword final_rwd=', self.rwd)
        else:
            self.rwd = 0

