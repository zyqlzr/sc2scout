import gym

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
    def __init__(self, env, max_step=4000, min_distance_allowed=5):
        super(TerminalWrapper, self).__init__(env)
        self._max_step = max_step
        self._min_distance_allowed = min_distance_allowed

    def _reset(self):
        obs = self.env._reset()
        return obs

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        scout = self.env.unwrapped.scout()
        enemy = self.env.unwrapped.enemy()
        scout_health = scout.float_attr.health
        max_health = scout.float_attr.health_max

        if scout_health < float(max_health)/5:
            return True

        if self.env.unwrapped.view_enemy() and enemy:
            if self.env.unwrapped._unit_dist(scout,enemy) < self._min_distance_allowed:
                return True

        return done
