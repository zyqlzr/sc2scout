
class ImgFeatExtractor(FeatureExtractor):
    def __init__(self, compress_width, channel_num, reverse=False):
        self.env = None
        self._compress_width = compress_width
        self._channel_num = channel_num
        self._reverse = reverse
        self._map_size = None
        self._x_radius = None
        self._y_radius = None
        self._x_per_unit = None
        self._y_per_unit = None

    def reset(self, env):
        self._map_size = env.unwrapped.map_size()
        self._x_radius = self._map_size[0] / 2
        self._y_radius = self._map_size[1] / 2
        self._x_per_unit = self._map_size[0] / self._compress_width
        self._y_per_unit = self._map_size[1] / self._compress_width

    def pos_2_2d(self, pos_x, pos_y):
        if self._reverse:
            if pos_x > self._x_radius:
                pos_x = self._x_radius - abs(x - self._x_radius)
            else:
                pos_x = self._x_radius + abs(x - self._x_radius)

            if pos_y > self._y_radius:
                pos_y = self._y_radius - abs(pos_y - self._y_radius)
            else:
                pos_y = self._y_radius + abs(pos_y - self._y_radius)

        i = pos_x * self._x_per_unit
        j = pos_y * self._y_per_unit
        return i, j

