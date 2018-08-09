from enum import Enum, unique

@unique
class TripStatus(Enum):
    FORWARD = 0
    EXPLORE = 1
    BACKWORD = 2
    TERMINAL = 3

class TripCourse(object):
    def __init__(self, home_pos, target_pos, target_range, step_num):
        self._home = home_pos
        self._target = target_pos
        self._target_range = target_range
        self._target_step = step_num

    def reset(self):
        x_radius = self._target_range[0] / 2
        y_radius = self._target_range[1] / 2
        self._x_low = self._target[0] - x_radius
        self._x_high = self._target[0] + x_radius
        self._y_low = self._target[1] - y_radius
        self._y_high = self._target[1] + y_radius
        print('target_range({},{}) <-> ({}, {})'.format(
              self._x_low, self._y_low, self._x_high, self._y_high))

        self._hx_low = self._home[0] - x_radius
        self._hx_high = self._home[0] + x_radius
        self._hy_low = self._home[1] - y_radius
        self._hy_high = self._home[1] + y_radius
        print('home_range({},{}) <-> ({}, {})'.format(
              self._hx_low, self._hy_low, self._hx_high, self._hy_high))

        self._status = TripStatus.FORWARD
        self._curr_step = 0

    def check_status(self, pos):
        if self._status == TripStatus.FORWARD:
            if self._in_target_range(pos):
                self._status = TripStatus.EXPLORE
        elif self._status == TripStatus.EXPLORE:
            self._curr_step += 1
            if self._curr_step == self._target_step:
                self._status = TripStatus.BACKWORD
        elif self._status == TripStatus.BACKWORD:
            if self._in_home_range(pos):
                self._status = TripStatus.TERMINAL
        else:
            print("Trip Terminal")
        return self._status

    def status(self):
        return self._status
 
    def _in_home_range(self, pos):
        if pos[0] > self._hx_low:
            return False
        elif pos[0] < self._hx_high:
            return False
        elif pos[1] > self._hy_low:
            return False
        elif pos[1] < self._hy_high:
            return False
        else:
            return True

    def _in_target_range(self, pos):
        if pos[0] > self._x_high:
            return False
        elif pos[0] < self._x_low:
            return False
        elif pos[1] > self._y_high:
            return False
        elif pos[1] < self._y_low:
            return False
        else:
            return True


