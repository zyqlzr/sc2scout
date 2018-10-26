import numpy as np
from gym.spaces import Box

from sc2scout.wrapper.feature.img_local_feat_extractor import ImgLocalFeatExtractor
from sc2scout.wrapper.util.dest_range import DestRange
import sc2scout.envs.scout_macro as sm

GLOBAL_CHANNEL = 5
MAX_UNIT_NUM = 200.

class FullGameLocalImgV2(ImgLocalFeatExtractor):
    def __init__(self, compress_width, local_range, target_range):
        super(FullGameLocalImgV2, self).__init__(compress_width, local_range)
        self._channel_num = GLOBAL_CHANNEL
        self._target_range = target_range
        self._target = None

    def reset(self, env):
        super(FullGameLocalImgV2, self).reset(env)
        self._target = DestRange(env.unwrapped.enemy_base(),
                                 dest_range=self._target_range)

    def extract(self, env, obs):
        image = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        scout = env.unwrapped.scout()
        if self._target.in_range((scout.float_attr.pos_x,
                                  scout.float_attr.pos_y)):
            neutrals, enemys = self.unit_dispatch(obs)
            channel_base = self.enemy_channel(enemys, image, 0)
            channel_base = self.neutral_channel(neutrals, image, channel_base)
            #print('local image: channel total number=', channel_base)
            #print('image=', image)
        return image

    def obs_space(self):
        low = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        high = np.ones([self._compress_width, self._compress_width, self._channel_num])
        return Box(low, high)

    def enemy_channel(self, enemys, image, channel_base):
        #air_set = []
        #base_set = []
        #building_set = []
        #other_set = []
        for u in enemys:
            u_pos = (u.float_attr.pos_x, u.float_attr.pos_y)
            if not self.check_in_range(u_pos):
                continue

            i, j = self.pos_2_2d(u_pos)
            #print('enemy coordinate={},{}'.format(i, j))
            if u.unit_type in sm.COMBAT_AIR_UNITS:
                image[i, j, channel_base + 0] += (1.0 / MAX_UNIT_NUM)
                #print('enemy air:[{},{},{}] = {}'.format(
                #      i, j, channel_base, image[i, j, channel_base]))
                #air_set.append((i, j))
            elif u.unit_type in sm.BASE_UNITS:
                image[i, j, channel_base + 1] = 1.0
                #base_set.append((i, j))
            elif u.unit_type in sm.BUILDING_UNITS:
                image[i, j, channel_base + 2] = (1.0 / MAX_UNIT_NUM)
                #building_set.append((i, j))
            else:
                image[i, j, channel_base + 3] += (1.0 / MAX_UNIT_NUM)
                #print('enemy unit:[{},{},{}] = {}'.format(
                #      i, j, channel_base + 1, image[i, j, channel_base + 1]))
                #other_set.append((i, j))
        #print('FullGameLocalImgV2 airs=', air_set)
        #print('FullGameLocalImgV2 bases=', base_set)
        #print('FullGameLocalImgV2 buildings=', building_set)
        #print('FullGameLocalImgV2 others=', other_set)
        return channel_base + 4

    def neutral_channel(self, neutrals, image, channel_base):
        for u in neutrals:
            u_pos = (u.float_attr.pos_x, u.float_attr.pos_y)
            if not self.check_in_range(u_pos):
                continue

            i, j = self.pos_2_2d(u_pos)
            image[i, j, channel_base + 0] += (1.0 / MAX_UNIT_NUM)
        return channel_base + 1

    def unit_dispatch(self, obs):
        units = obs.observation['units']
        neutrals = []
        enemys = []
        for u in units:
            if u.int_attr.alliance == sm.AllianceType.NEUTRAL.value:
                neutrals.append(u)
            elif u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        return neutrals, enemys

