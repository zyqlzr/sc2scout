import numpy as np
from gym.spaces import Box

from sc2scout.wrapper.feature.img_feat_extractor import ImgFeatExtractor
from sc2scout.wrapper.util.map_scan import MapScan
from sc2scout.wrapper.util.trip_status import TripStatus, TripCourse
import sc2scout.envs.scout_macro as sm

GLOBAL_CHANNEL = 24

class ScoutGlobalImgFeatureV1(ImgFeatExtractor):
    def __init__(self, compress_width, range_width, explore_step, reverse):
        super(ScoutGlobalImgFeatureV1, self).__init__(compress_width, reverse)
        self._channel_num = GLOBAL_CHANNEL
        self._range_width = range_width
        self._explore_step = explore_step
        self._map_scan = MapScan(compress_width)
        self._status = None

    def reset(self, env):
        super(ScoutGlobalImgFeatureV1, self).reset(env)
        self._map_scan = MapScan(self._compress_width)
        self._init_status(env)
        print('GlobalImgFeatureV1 radius=({},{}), per_unit=({},{})'.format(
              self._x_radius, self._y_radius, self._x_per_unit, self._y_per_unit))

    def _init_status(self, env):
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        x_range = self._x_per_unit * self._range_width
        y_range = self._y_per_unit * self._range_width
        self._status = TripCourse(home_pos, enemy_pos, 
                                  (x_range, y_range), self._explore_step)
        self._status.reset()

    def extract(self, env, obs):
        owners, neutrals, enemys = self.unit_dispatch(obs)
        image = np.zeros([self._compress_width, self._compress_width, self._channel_num])
        #print('image shape=', image.shape)
        channel_base = self.home_pos_channel(env, image, 0)
        channel_base = self.target_pos_channel(env, image, channel_base)
        channel_base = self.scout_attr_channel(env, image, channel_base)
        channel_base = self.nertral_attr_channel(neutrals, image, channel_base)
        channel_base = self.owner_attr_channel(owners, image, channel_base)
        channel_base = self.enemy_attr_channel(enemys, image, channel_base)
        channel_base = self.scout_scan_channel(env, image, channel_base)
        channel_base = self.scout_status_channel(env, image, channel_base)
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
        #print("home coordinate({}, {}), home={}".format(i, j, home))
        image[i, j, channel_num] += 1
        return channel_num + 1

    def target_pos_channel(self, env, image, channel_num):
        target = env.unwrapped.enemy_base()
        i, j = self.pos_2_2d(target[0], target[1])
        #print("target coordinate({}, {}), target={}".format(i, j, target))
        image[i, j, channel_num] += 1
        return channel_num + 1

    def scout_attr_channel(self, env, image, channel_base):
        scout = env.unwrapped.scout()
        i, j = self.pos_2_2d(scout.float_attr.pos_x, scout.float_attr.pos_y)
        image[i, j, channel_base + 0] = 1 # scout in this position
        #print("scout coordinate({}, {}), scout=({},{})".format(i, j, 
        #    scout.float_attr.pos_x, scout.float_attr.pos_y))
        if scout.bool_attr.is_flying:
            image[i, j, channel_base + 1] = 1  # flying
        else:
            image[i, j, channel_base + 1] = 0  # flying
        image[i, j, channel_base + 2] = scout.float_attr.facing
        image[i, j, channel_base + 3] = scout.float_attr.radius
        image[i, j, channel_base + 4] = scout.float_attr.health
        image[i, j, channel_base + 5] = scout.float_attr.health_max
        #print("scout image attr:", image[i,j,:])
        return channel_base + 6

    def scout_scan_channel(self, env, image, channel_base):
        scout = env.unwrapped.scout()
        i, j = self.pos_2_2d(scout.float_attr.pos_x, scout.float_attr.pos_y)
        self._map_scan.scan_pos(i, j)
        image[:,:,channel_base] = self._map_scan.scan_map()
        '''
        print('row=', i, ';scout_scan:', image[i,:,channel_base],
              ';map_scan:', self._map_scan.scan_map()[i])
        print('column=', j, 'scout_scan:', image[:,j,channel_base],
              ';map_scan:', self._map_scan.scan_map()[:,j])
        '''
        return channel_base + 1

    def scout_status_channel(self, env, image, channel_base):
        ones  = np.ones([self._compress_width, self._compress_width])
        zeros  = np.zeros([self._compress_width, self._compress_width])
        if self._status.status() == TripStatus.FORWARD:
            image[:, :, channel_base + 0] = ones
            image[:, :, channel_base + 1] = zeros
            image[:, :, channel_base + 2] = zeros
            #print('forward feature, pos=({},{}), channel_base={},channel_list={}'.format(
            #      i, j, channel_base, image[i, j, :]))
        elif self._status.status() == TripStatus.EXPLORE:
            image[:, :, channel_base + 0] = zeros 
            image[:, :, channel_base + 1] = ones
            image[:, :, channel_base + 2] = zeros
            #print('explore feature, pos=({},{}), channel_base={},channel_list={}'.format(
            #      i, j, channel_base, image[i, j, :]))
        elif self._status.status() == TripStatus.BACKWORD:
            image[:, :, channel_base + 0] = zeros
            image[:, :, channel_base + 1] = zeros
            image[:, :, channel_base + 2] = ones
            #print('backward feature, pos=({},{}), channel_base={},channel_list={}'.format(
            #      i, j, channel_base, image[i, j, :]))
        else:
            image[:, :, channel_base + 0] = zeros
            image[:, :, channel_base + 1] = zeros
            image[:, :, channel_base + 2] = zeros
            #print('terminal feature, pos=({},{}), channel_base={},channel_list={}'.format(
            #      i, j, channel_base, image[i, j, :]))
        return channel_base + 3

    def nertral_attr_channel(self, neutrals, image, channel_base):
        nertral_count = 0
        resource_count = 0
        for u in neutrals:
            i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
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

    def owner_attr_channel(self, owners, image, channel_base):
        for u in owners:
            i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
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

    def enemy_attr_channel(self, enemys, image, channel_base):
        for u in enemys:
            i, j = self.pos_2_2d(u.float_attr.pos_x, u.float_attr.pos_y)
            #print('enemy coordinate={},{}'.format(i, j))
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


