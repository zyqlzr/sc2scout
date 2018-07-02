
class FeatureExtractor(object):
    def extract(self, env, obs):
        raise NotImplementedError

    def to_array(self):
        raise NotImplementedError

class ScoutAvoidImgFeature(FeatureExtractor):
    def __init__(self):
        super(ScoutAvoidImgFeature, self).__init__()

    def extract(self, env, obs):
        pass

    def to_array(self):
        pass


class ScoutAvoidVecFeature(FeatureExtractor):
    def __init__(self):
        super(ScoutAvoidImgFeature, self).__init__()


    def extract(self, env, obs):
        pass

    def to_array(self):
        pass


