
class Reward(object):
    def __init__(self, w):
        self.w = w
        self.rwd = 0

    def reset(self, obs, env):
        raise NotImplementedError

    def compute_rwd(self, obs, reward, done, env):
        raise NotImplementedError

