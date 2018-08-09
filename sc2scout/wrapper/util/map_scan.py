import numpy as np
from sc2scout.wrapper.feature.img_feat_extractor import ImgFeatExtractor
import math

class MapScan(object):
    def __init__(self, size):
        self._shape = (size, size)
        self._map = np.zeros(self._shape, dtype=np.int8)

    def scan_pos(self, i, j):
        if 0 == self._map[i, j]:
            self._map[i, j] = 1
            return True
        else:
            return False

    def scan_map(self):
        return self._map

class TargetMapScan(ImgFeatExtractor):
    def __init__(self, compress_width, center, target_range):
        super(TargetMapScan, self).__init__(compress_width)
        self._center = center
        self._range = math.floor(target_range / 2)
        self._shape = (self._range, self._range)
        self._map = np.zeros(self._shape, dtype=np.int8)

    def reset(self, env):
        super(TargetMapScan, self).reset(env)
        self._c_i, self._c_j = self.pos_2_2d(self._center[0], self._center[1])

        radius = (self._x_per_unit * self._range) / 2
        min_pos = (self._center[0] - radius, self._center[1] - radius)
        max_pos = (self._center[0] + radius, self._center[1] + radius)
        self._min_i, self._min_j = self.pos_2_2d(min_pos[0], min_pos[1])
        self._max_i, self._max_j = self.pos_2_2d(max_pos[0], max_pos[1])
        #print("TargetMapScan radius={},min=({},{}),max=({},{})".format(
        #      radius, self._min_i, self._min_j, self._max_i, self._max_j))

    def scan_pos(self, pos_x, pos_y):
        i, j = self.pos_2_2d(pos_x, pos_y)
        if i < self._min_i or i > self._max_i:
            return False
        if j < self._min_j or j > self._max_j:
            return False

        target_i = i - self._min_i
        target_j = j - self._min_j
        assert(target_i <= self._range)
        assert(target_j <= self._range)
        if 0 == self._map[target_i, target_j]:
            self._map[target_i, target_j] = 1
            #print("TargetMapScan Hit map_pos=({},{}) pos=({},{}), target_pos=({},{})".format(
            #      pos_x, pos_y, i, j, target_i, target_j))
            return True
        else:
            #print("TargetMapScan Miss map_pos=({},{}) pos=({},{}), target_pos=({},{})".format(
            #      pos_x, pos_y, i, j, target_i, target_j))
            return False

    def extract(self, env, obs):
        raise NotImplementedError

    def obs_space(self):
        raise NotImplementedError


if __name__ == '__main__':
    arr = np.zeros((5, 5, 2), dtype=np.int8)
    print(arr[:,:,0])
    arr[:,:,0] = np.ones((5,5), dtype=np.int8)
    print(arr[:,:,0])
    print(arr)
