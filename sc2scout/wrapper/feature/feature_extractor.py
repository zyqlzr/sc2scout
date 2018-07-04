
class FeatureExtractor(object):
    def reset(self, env):
        raise NotImplementedError

    def extract(self, env, obs):
        raise NotImplementedError

    def obs_space(self):
        raise NotImplementedError


