from sc2scout.wrapper.feature.feature_extractor import FeatureExtractor
import math

class ImgFeatExtractor(FeatureExtractor):
    def __init__(self, compress_width, reverse=False):
        self.env = None
        self._compress_width = compress_width
        self._reverse = False
        self._map_size = None
        self._x_radius = None
        self._y_radius = None
        self._x_per_unit = None
        self._y_per_unit = None

    def reset(self, env):
        map_size = env.unwrapped.map_size()
        self._map_size = (float(map_size[0]), float(map_size[1]))
        self._x_radius = self._map_size[0] / 2
        self._y_radius = self._map_size[1] / 2
        self._x_per_unit = self._map_size[0] / self._compress_width
        self._y_per_unit = self._map_size[1] / self._compress_width
        print("map_size={},x_rad={},y_rad={},x_per={},y_per={}".format(
              self._map_size, self._x_radius, self._y_radius,
              self._x_per_unit, self._y_per_unit))
        self._reverse = env.unwrapped.judge_reverse()

    def pos_2_2d(self, pos_x, pos_y):
        if self._reverse:
            if pos_x > self._x_radius:
                pos_x = self._x_radius - abs(pos_x - self._x_radius)
            else:
                pos_x = self._x_radius + abs(pos_x - self._x_radius)

            if pos_y > self._y_radius:
                pos_y = self._y_radius - abs(pos_y - self._y_radius)
            else:
                pos_y = self._y_radius + abs(pos_y - self._y_radius)

        i = math.floor(pos_x / self._x_per_unit)
        j = math.floor(pos_y / self._y_per_unit)
        return i, j

    def pos_2_2d_local(self, pos_x, pos_y, cx, cy):
        pos_x = (pos_x - cx) + self._x_radius
        pos_y = (pos_y - cy) + self._y_radius
        i = math.floor(pos_x / self._x_per_unit)
        j = math.floor(pos_y / self._y_per_unit)
        return i, j

if __name__ == "__main__":
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

    env1 = Env(True)
    imgft1 = ImgFeatExtractor(32, True)
    imgft1.reset(env1)
    print(imgft1.pos_2_2d(59.5, 27.5))
    print(imgft1.pos_2_2d(28.5, 60.5))

    env2 = Env(False)
    imgft2 = ImgFeatExtractor(32, False)
    imgft2.reset(env2)
    print(imgft2.pos_2_2d(59.5, 27.5))
    print(imgft2.pos_2_2d(28.5, 60.5))
    print(imgft2.pos_2_2d(44.0, 48.0))

