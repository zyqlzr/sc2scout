import numpy as np
from gym.spaces import Box

from sc2scout.wrapper.feature.img_feat_extractor import ImgFeatExtractor
import sc2scout.envs.scout_macro as sm

LOCAL_CHANNEL = 8

class ScoutLocalImgFeatureV2(ImgFeatExtractor):
    def __init__(self, compress_width, local_width, reverse):
        super(ScoutLocalImgFeatureV2, self).__init__(compress_width, reverse)
        self._local_width = local_width
        self._channel_num = LOCAL_CHANNEL

    def reset(self, env):
        super(ScoutLocalImgFeatureV2, self).reset(env)
        local_x = self._x_per_unit * self._local_width
        local_y = self._y_per_unit * self._local_width
        self._x_radius = local_x / 2
        self._y_radius = local_y / 2
        print('Local Feature, local({}, {}), radius({}, {}), per_unit=({},{})'.format(
            local_x, local_y, self._x_radius, self._y_radius, 
            self._x_per_unit, self._y_per_unit))

    def extract(self, env, obs):
        owners, neutrals, enemys = self.unit_dispatch(obs, env)
        image = np.zeros([self._local_width, self._local_width, self._channel_num])
        channel_base = self.scout_attr_channel(env, image, 0)
        channel_base = self.nertral_attr_channel(neutrals, image, channel_base, env)
        channel_base = self.enemy_attr_channel(enemys, image, channel_base, env)
        return image

    def obs_space(self):
        low = np.zeros([self._local_width, self._local_width, self._channel_num])
        high = np.ones([self._local_width, self._local_width, self._channel_num])
        return Box(low, high)

    def check_in_range(self, pos_x, pos_y, env):
        scout = env.unwrapped.scout()
        x_low = scout.float_attr.pos_x - self._x_radius
        x_high = scout.float_attr.pos_x + self._x_radius
        y_low = scout.float_attr.pos_y - self._y_radius
        y_high = scout.float_attr.pos_y + self._y_radius
        if pos_x > x_high or pos_x < x_low:
            return False
        if pos_y > y_high or pos_y < y_low:
            return False
        return True

    def center_pos(self, env):
        scout = env.unwrapped.scout()
        cx = scout.float_attr.pos_x
        cy = scout.float_attr.pos_y
        return cx, cy

    def scout_attr_channel(self, env, image, channel_base):
        scout = env.unwrapped.scout()
        cx, cy = self.center_pos(env)
        i, j = self.pos_2_2d_local(cx, cy, cx, cy)
        #print("local_feat: scout coordinate({}, {}), pos=({},{})".format(i, j, cx, cy))
        image[i, j, channel_base] = scout.float_attr.health / scout.float_attr.health_max
        #print("scout image attr:", image[i,j,:])
        return channel_base + 1

    def nertral_attr_channel(self, neutrals, image, channel_base, env):
        if 0 == len(neutrals):
            return channel_base + 2

        nertral_count = 0
        resource_count = 0
        cx, cy = self.center_pos(env)
        for u in neutrals:
            i, j = self.pos_2_2d_local(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy)
            #print("nertral coordinate({}, {}), pos=({},{}), c=({},{})".format(i, j, 
            #    u.float_attr.pos_x, u.float_attr.pos_y, cx, cy))
            if u.unit_type in sm.MINERAL_UNITS or u.unit_type in sm.VESPENE_UNITS:
                resource_count += 1
                image[i, j, channel_base + 0] += 1
                #print('**resrouce unit=', u.tag, '; resource type=', u.unit_type)
            else:
                nertral_count += 1
                image[i, j, channel_base + 1] += 1
                #print('--neutral unit=', u.tag, '; enutral type=', u.unit_type)
        #print("---nertral_count---,", nertral_count)
        #print("***nertral_count***,", resource_count)
        return channel_base + 2

    def enemy_attr_channel(self, enemys, image, channel_base, env):
        if 0 == len(enemys):
            return channel_base + 5

        cx, cy = self.center_pos(env)
        for u in enemys:
            i, j = self.pos_2_2d_local(u.float_attr.pos_x, u.float_attr.pos_y, cx, cy)
            #print("local_feat: enemy coordinate({}, {}), pos=({},{}), c=({},{})".format(
            #      i, j, u.float_attr.pos_x, u.float_attr.pos_y, cx, cy))
            if u.unit_type in sm.BASE_UNITS:
                image[i, j, channel_base + 0] += 1
            elif u.unit_type in sm.BUILDING_UNITS:
                image[i, j, channel_base + 1] += 1
            elif u.unit_type in sm.COMBAT_AIR_UNITS:
                image[i, j, channel_base + 2] += 1
            elif u.unit_type in sm.COMBAT_UNITS:
                image[i, j, channel_base + 3] += 1
            else:
                image[i, j, channel_base + 4] += 1
        return channel_base + 5

    def unit_dispatch(self, obs, env):
        units = obs.observation['units']
        owners = []
        neutrals = []
        enemys = []
        for u in units:
            if not self.check_in_range(u.float_attr.pos_x, u.float_attr.pos_y, env):
                continue

            if u.int_attr.alliance == sm.AllianceType.SELF.value:
                owners.append(u)
            elif u.int_attr.alliance == sm.AllianceType.NEUTRAL.value:
                neutrals.append(u)
            elif u.int_attr.alliance == sm.AllianceType.ENEMY.value:
                enemys.append(u)
            else:
                continue

        #print('owners={},neutrals={}, enemys={}'.format(
        #    len(owners), len(neutrals), len(enemys)))
        return owners, neutrals, enemys


