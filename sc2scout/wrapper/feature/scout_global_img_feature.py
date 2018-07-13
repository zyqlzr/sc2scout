from sc2scout.wrapper.feature.img_feat_extractor import ImgFeatExtractor 

GLOBAL_CHANNEL = 20

class ScoutGlobalImgFeature(ImgFeatExtractor):
    def __init__(self, compress_width, reverse):
        super(ScoutGlobalImgFeature, self).__init__(compress_width, GLOBAL_CHANNEL, reverse)

    def reset(self, env):
        super(ScoutGlobalImgFeature, self).reset(env)

    def extract(self, env, obs):
        image = np.zeros([self._compress_width, self._compress_width, self._channel_num])

    def obs_space(self):
        pass

    def home_pos_channel(self, env, image, channel_num):
        home = env.unwrapped.owner_base()
        i, j = self.pos_2_2d(home[0], home[1])
        image[i, j, channel_num] = 1
        return i, j

    def target_pos_channel(self, env):
        target = env.unwrapped.enemy_base()
        i, j = self.pos_2_2d(target[0], target[1])
        return i, j

    def scout_pos_channel(self, env):
        scout = env.unwrapped.scout()
        i, j = self.pos_2_2d(scout.float_attr.pos_x, scout.float_attr.pos_y)
        return i, j

    '''
    def scout_attr_channel(self, env)
    '''

