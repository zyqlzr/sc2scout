import numpy as np
from gym.spaces import Box

from sc2scout.wrapper.feature.img_local_feat_extractor import ImgLocalFeatExtractor
from sc2scout.wrapper.util.dest_range import DestRange
import sc2scout.envs.scout_macro as sm

GLOBAL_CHANNEL = 2
MAX_UNIT_NUM = 100.

class FullGameLocalImgV1(ImgLocalFeatExtractor):
    def __init__(self, compress_width, local_range, target_range):
        super(FullGameLocalImgV1, self).__init__(compress_width, local_range)
        self._channel_num = GLOBAL_CHANNEL
        self._target_range = target_range
        self._target = None

    def reset(self, env):
        super(FullGameLocalImgV1, self).reset(env)
        self._target = DestRange(env.unwrapped.enemy_base(), 
                                 dest_range=self._target_range)

    def extract(self, env, obs):
        image = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        scout = env.unwrapped.scout()
        if self._target.in_range((scout.float_attr.pos_x,
                                  scout.float_attr.pos_y)):
            enemys = self.unit_dispatch(obs)
            channel_base = self.enemy_channel(enemys, image, 0)
            print('local image: channel total number=', channel_base)
            #print('image=', image)
        else:
            print('empty local image')
        return image

    def obs_space(self):
        low = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        high = np.ones([self._compress_width, self._compress_width, self._channel_num])
        return Box(low, high)

    def enemy_channel(self, enemys, image, channel_base):
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
            else:
                image[i, j, channel_base + 1] += (1.0 / MAX_UNIT_NUM)
                #print('enemy unit:[{},{},{}] = {}'.format(
                #      i, j, channel_base + 1, image[i, j, channel_base + 1]))
        return channel_base + 2

    def unit_dispatch(self, obs):
        units = obs.observation['units']
        enemys = []
        for u in units:
            if u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        return enemys

