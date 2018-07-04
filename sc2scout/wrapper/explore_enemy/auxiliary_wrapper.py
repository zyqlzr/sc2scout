import gym
from sc2scout.envs import scout_macro as sm
from sc2scout.wrapper.util.dest_range import DestRange

class SkipFrame(gym.Wrapper):
    """ Skip specific no. of frames by executing noop or specified actions """
    def __init__(self, env, skip_count=3, repeat_action=None):
        super(SkipFrame, self).__init__(env)
        self.skip_count = skip_count
        if repeat_action is None:
            self.repeated_action = self.metadata['action.noop']
        else:
            self.repeated_action = repeat_action

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        #self._print_scout_pos()
        if done:
            return obs, rwd, done, info
        for _ in range(self.skip_count):
            obs_, rwd_, done_, info_ = self.env._step(self.repeated_action)
            #self._print_scout_pos()
            obs = self._process_skip_obs(obs_, obs)
            rwd = self._process_skip_rwd(rwd_, rwd)
            info = self._process_skip_info(info_, info)
            if done_:
                return obs, rwd, done_, info
            done = done_
        return obs, rwd, done, info

    def _process_skip_rwd(self, skip_rwd, rwd):
        return skip_rwd

    def _process_skip_obs(self, skip_obs, obs):
        return skip_obs

    def _process_skip_info(self, skip_info, info):
        return skip_info

    def _print_scout_pos(self):
        scout = self.env.unwrapped.scout()
        print('skip scout pos={}'.format((scout.float_attr.pos_x, 
                                    scout.float_attr.pos_y)))


class TerminalWrapper(gym.Wrapper):
    def __init__(self, env, max_step=4000):
        super(TerminalWrapper, self).__init__(env)
        self._max_step = max_step
        self._enemy_base = None

    def _reset(self):
        obs = self.env._reset()
        self._enemy_base = DestRange(self.env.unwrapped.enemy_base())
        return obs

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        scout = self.env.unwrapped.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)

        self._enemy_base.check_enter(pos)
        self._enemy_base.check_hit(pos)
        self._enemy_base.check_leave(pos)
        if self._enemy_base.hit:
            print('***episode terminal while scout hit ***')
            return True

        if self._enemy_base.enter and self._enemy_base.leave:
            print('***episode terminal while scout enter and leave***')
            return True
        return done

class RoundTripTerminalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RoundTripTerminalWrapper, self).__init__(env)
        self._enemy_base = None
        self._home_base = None
        self._back = False

    def _reset(self):
        obs = self.env._reset()
        self._enemy_base = DestRange(self.env.unwrapped.enemy_base())
        self._home_base = DestRange(self.env.unwrapped.owner_base())
        self._back = False
        return obs

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        info = self._judge_course(obs)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if self._back and self._home_base.enter:
            return True
        else:
            return done

    def _judge_course(self, obs):
        scout = self.env.unwrapped.scout()
        pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        if not self._back:
            self._enemy_base.check_enter(pos)
            self._enemy_base.check_hit(pos)
            self._enemy_base.check_leave(pos)
        else:
            self._home_base.check_enter(pos)
            self._home_base.check_hit(pos)
            self._home_base.check_leave(pos)

        if self._enemy_base.enter and self._enemy_base.leave:
            if not self._back:
                self._back = True

        return self._back
