from sc2scout.envs import scout_macro as sm

MIN_DIST_ARANGE = 2
DEST_RANGE = 8

class DestRange(object):
    def __init__(self, center_pos, dest_range=DEST_RANGE, hit_range=MIN_DIST_ARANGE):
        self._center = center_pos
        self._dest_range = dest_range
        self._hit_range = hit_range

        self.enter = False
        self.leave = False
        self.hit = False

    def check_hit(self, pos):
        if self.hit:
            return
        dist =  sm.calculate_distance(pos[0], pos[1],
                                     self._center[0], self._center[1])
        if dist < self._hit_range:
            self.hit = True

    def check_enter(self, pos):
        if self.enter:
            return
        dist = sm.calculate_distance(pos[0], pos[1],
                                     self._center[0], self._center[1])
        if dist < self._dest_range:
            self.enter = True

    def check_leave(self, pos):
        if not self.enter:
            return

        if self.leave:
            return
        dist = sm.calculate_distance(pos[0], pos[1],
                                     self._center[0], self._center[1])
        if dist > self._dest_range:
            self.leave = True

    def in_range(self, pos):
        dist = sm.calculate_distance(pos[0], pos[1],
                                     self._center[0], self._center[1])
        if dist > DEST_RANGE:
            return False
        else:
            return True 



