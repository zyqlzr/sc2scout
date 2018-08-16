import gym
from sc2scout.wrapper.util.trip_status import TripStatus, TripCourse
from sc2scout.wrapper.util.dest_range import DestRange

class TargetTerminalWrapper(gym.Wrapper):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(TargetTerminalWrapper, self).__init__(env)
        self._status = None
        self._compress_width = compress_width
        self._range_width = range_width
        self._explore_step = explore_step

    def _reset(self):
        obs = self.env._reset()
        map_size = self.env.unwrapped.map_size()
        x_per_unit = map_size[0] / self._compress_width
        y_per_unit = map_size[1] / self._compress_width
        home_pos = self.env.unwrapped.owner_base()
        enemy_pos = self.env.unwrapped.enemy_base()
        x_range = x_per_unit * self._range_width
        y_range = y_per_unit * self._range_width
        self._status = TripCourse(home_pos, enemy_pos, 
                                  (x_range, y_range), self._explore_step)
        self._status.reset()
        self._home_base = DestRange(self.env.unwrapped.owner_base())
        return obs

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        survive = self.env.unwrapped.scout_survive()
        if not survive:
            return True

        scout = self.env.unwrapped.scout()
        status = self._status.check_status((scout.float_attr.pos_x, scout.float_attr.pos_y))
        if status == TripStatus.TERMINAL:
            return True
        elif status == TripStatus.BACKWORD:
            if self._home_base.in_range((scout.float_attr.pos_x, scout.float_attr.pos_y)):
                print("backward return_HOME,", status)
                return True
        return done

class TargetTerminalWrapperV1(gym.Wrapper):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(TargetTerminalWrapperV1, self).__init__(env)

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        survive = self.env.unwrapped.scout_survive()
        if not survive:
            return True
       
        return done

