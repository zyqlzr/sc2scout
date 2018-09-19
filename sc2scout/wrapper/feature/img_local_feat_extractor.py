from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
import math

class ImgLocalFeatExtractor(FeatureExtractor):
    def __init__(self, compress_width, local_range):
        self.env = None
        self._compress_width = compress_width
        self._local_range = local_range
        self._x_radius = None
        self._y_radius = None
        self._local_per_unit = None
        self._reverse = False
        self._map_size = None

    def reset(self, env):
        self._map_size = env.unwrapped.map_size()
        self._x_radius = self._map_size[0] / 2
        self._y_radius = self._map_size[1] / 2
        self._local_per_unit = 2 * self._local_range / self._compress_width
        print("map_size={},x_rad={},y_rad={},local_per_unit={}, local_range={}".format(
              self._map_size, self._x_radius, self._y_radius,
              self._local_per_unit, self._local_range))
        self._reverse = env.unwrapped.judge_reverse()
        self.env = env.unwrapped

    def pos_2_2d(self, pos):
        scout = self.env.scout()
        scout_pos = self._trans_pos(scout.float_attr.pos_x, scout.float_attr.pos_y)
        pos = self._trans_pos(pos[0], pos[1])

        x_len = pos[0] - scout_pos[0] + self._local_range
        y_len = pos[1] - scout_pos[1] + self._local_range
        i = math.floor(x_len / self._local_per_unit)
        j = math.floor(y_len / self._local_per_unit)
        return i, j

    def check_in_range(self, pos):
        scout = self.env.scout()
        scout_pos = self._trans_pos(scout.float_attr.pos_x, scout.float_attr.pos_y)
        pos = self._trans_pos(pos[0], pos[1])
        #print('trans: scout_pos=', scout_pos, '; input=', pos)
        min_x = scout_pos[0] - self._local_range
        min_y = scout_pos[1] - self._local_range
        max_x = scout_pos[0] + self._local_range
        max_y = scout_pos[1] + self._local_range
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x = max_x if max_x < self._map_size[0] else self._map_size[0]
        max_y = max_y if max_y < self._map_size[1] else self._map_size[1]

        if pos[0] >= max_x or pos[0] < min_x:
            return False

        if pos[1] >= max_y or pos[1] < min_y:
            return False

        return True

    def _trans_pos(self, pos_x, pos_y):
        if self._reverse:
            if pos_x > self._x_radius:
                pos_x = self._x_radius - abs(pos_x - self._x_radius)
            else:
                pos_x = self._x_radius + abs(pos_x - self._x_radius)

            if pos_y > self._y_radius:
                pos_y = self._y_radius - abs(pos_y - self._y_radius)
            else:
                pos_y = self._y_radius + abs(pos_y - self._y_radius)
        return (pos_x, pos_y)

if __name__ == "__main__":
    class Scout:
        def __init__(self, x, y):
            self._x = x
            self._y = y
        @property
        def float_attr(self):
            return self
        @property
        def pos_x(self):
            return self._x
        @property
        def pos_y(self):
            return self._y

    class Env:
        def __init__(self, reverse):
            self._reverse = reverse

        @property
        def unwrapped(self):
            return self

        def map_size(self):
            return (88, 96)

        def judge_reverse(self):
            return self._reverse

        def scout(self):
            return Scout(32, 96)

    #env1 = Env(True)
    env1 = Env(False)
    imgft1 = ImgLocalFeatExtractor(32, 12)
    imgft1.reset(env1)

    pos_arr = []
    pos_arr.append((44, 44))
    pos_arr.append((20, 20))
    pos_arr.append((21, 94))

    for pos in pos_arr:
        if imgft1.check_in_range(pos):
            print(imgft1.pos_2_2d(pos))
        else:
            print("pos is out of range,", pos)


