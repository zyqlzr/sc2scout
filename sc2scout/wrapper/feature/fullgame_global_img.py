import numpy as np
from gym.spaces import Box

from sc2scout.wrapper.feature.img_feat_extractor import ImgFeatExtractor
import sc2scout.envs.scout_macro as sm

GLOBAL_CHANNEL = 14
MAX_UNIT_NUM = 100.

class FullGameGlobalImg(ImgFeatExtractor):
    def __init__(self, compress_width):
        super(FullGameGlobalImg, self).__init__(compress_width)
        self._channel_num = GLOBAL_CHANNEL

    def reset(self, env):
        super(FullGameGlobalImg, self).reset(env)
        print('FullGameGlobalImg radius=({},{}), per_unit=({},{})'.format(
              self._x_radius, self._y_radius, self._x_per_unit, self._y_per_unit))

    def extract(self, env, obs):
        owners, neutrals, enemys = self.unit_dispatch(obs)
        image = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        '''update status'''
        scout = env.unwrapped.scout()

        channel_base = self.home_pos_channel(env, image, 0)
        channel_base = self.target_pos_channel(env, image, channel_base)
        channel_base = self.scout_attr_channel(env, image, channel_base)
        channel_base = self.nertral_attr_channel(neutrals, image, channel_base)
        channel_base = self.owner_attr_channel(owners, image, channel_base)
        channel_base = self.enemy_attr_channel(enemys, image, channel_base)
        #print('channel total number=', channel_base)
        #print('image=', image)
        return image

    def obs_space(self):
        low = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        high = np.ones([self._compress_width, self._compress_width, self._channel_num])
        return Box(low, high)

    def home_pos_channel(self, env, image, channel_num):
        home = env.unwrapped.owner_base()
        i, j = self.pos_2_2d(home[0], home[1])
        #print("fullgame home coordinate({}, {}), home={}".format(i, j, home))
        image[i, j, channel_num] += 1
        return channel_num + 1

    def target_pos_channel(self, env, image, channel_num):
        target = env.unwrapped.enemy_base()
        i, j = self.pos_2_2d(target[0], target[1])
        #print("fullgame target coordinate({}, {}), target={}".format(i, j, target))
        image[i, j, channel_num] += 1
        return channel_num + 1

    def scout_attr_channel(self, env, image, channel_base):
        scout = env.unwrapped.scout()
        i, j = self.pos_2_2d(scout.float_attr.pos_x, scout.float_attr.pos_y)
        image[i, j, channel_base + 0] = 1 # scout in this position
        #print("fullgame scout coordinate({}, {}), scout=({},{})".format(
        #       i, j, scout.float_attr.pos_x, scout.float_attr.pos_y))
        return channel_base + 1
 
    def nertral_attr_channel(self, neutrals, image, channel_base):
        resource_count = 0
        for u in neutrals:
            i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
            if u.unit_type in sm.MINERAL_UNITS or u.unit_type in sm.VESPENE_UNITS:
                resource_count += 1
                image[i, j, channel_base + 0] += (1.0 / MAX_UNIT_NUM)
                #print("nertral_attr:[{},{},{}]={} ".format(
                #      i, j, channel_base, image[i, j, channel_base]))
        #print("***resource_count***,", resource_count)
        return channel_base + 1

    def owner_attr_channel(self, owners, image, channel_base):
        for u in owners:
            i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
            if u.unit_type in sm.BASE_UNITS:
                image[i, j, channel_base + 0] += (1.0 / MAX_UNIT_NUM)
                #print("owner 0=", image[i, j, channel_base])
            elif u.unit_type in sm.BUILDING_UNITS:
                image[i, j, channel_base + 1] += (1.0 / MAX_UNIT_NUM)
                #print("owner 1=", image[i, j, channel_base + 1])
            elif u.unit_type in sm.COMBAT_AIR_UNITS:
                image[i, j, channel_base + 2] += (1.0 / MAX_UNIT_NUM)
                #print("owner 2=", image[i, j, channel_base + 2])
            elif u.unit_type in sm.COMBAT_UNITS:
                image[i, j, channel_base + 3] += (1.0 / MAX_UNIT_NUM)
                #print("owner 3=", image[i, j, channel_base + 3])
            else:
                image[i, j, channel_base + 4] += (1.0 / MAX_UNIT_NUM)
                #print("owner 4=", image[i, j, channel_base + 4])
            #print("owner_attr:[{},{},{}]={}".format(
            #      i, j, channel_base, image[i, j, channel_base:channel_base+5]))
        #print("***owner_count***,", len(owners))
        return channel_base + 5

    def enemy_attr_channel(self, enemys, image, channel_base):
        for u in enemys:
            i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
            #print('enemy coordinate={},{}'.format(i, j))
            if u.unit_type in sm.BASE_UNITS:
                image[i, j, channel_base + 0] += (1.0 / MAX_UNIT_NUM)
                #print("enemy 0=", image[i, j, channel_base])
            elif u.unit_type in sm.BUILDING_UNITS:
                image[i, j, channel_base + 1] += (1.0 / MAX_UNIT_NUM)
                #print("enemy 1=", image[i, j, channel_base + 1])
            elif u.unit_type in sm.COMBAT_AIR_UNITS:
                image[i, j, channel_base + 2] += (1.0 / MAX_UNIT_NUM)
                #print("enemy 2=", image[i, j, channel_base + 2])
            elif u.unit_type in sm.COMBAT_UNITS:
                image[i, j, channel_base + 3] += (1.0 / MAX_UNIT_NUM)
                #print("enemy 3=", image[i, j, channel_base + 3])
            else:
                image[i, j, channel_base + 4] += (1.0 / MAX_UNIT_NUM)
                #print("enemy 4=", image[i, j, channel_base + 4])
            #print("enemy_attr:[{},{},{}]={}".format(
            #      i, j, channel_base, image[i, j, channel_base:channel_base+5]))
        #print("***enemy_count***,", len(enemys))
        return channel_base + 5

    def unit_dispatch(self, obs):
        units = obs.observation['units']
        owners = []
        neutrals = []
        enemys = []
        for u in units:
            if u.int_attr.alliance == sm.AllianceType.SELF.value:
                owners.append(u)
            elif u.int_attr.alliance == sm.AllianceType.NEUTRAL.value:
                neutrals.append(u)
            elif u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        return owners, neutrals, enemys

