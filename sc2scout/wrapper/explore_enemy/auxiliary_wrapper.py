import gym
from sc2scout.envs import scout_macro as sm

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

MIN_DIST_ARANGE = 2

class TerminalWrapper(gym.Wrapper):
    def __init__(self, env, max_step=4000):
        super(TerminalWrapper, self).__init__(env)
        self._max_step = max_step

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        if self._check_arrived(obs):
            print('***episode terminal while scout arrived***')
            return True
        return done

    def _check_arrived(self, obs):
        scout = self.env.unwrapped.scout()
        enemy_base = self.env.unwrapped.enemy_base()
        dist =  sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     enemy_base[0], enemy_base[1])
        return dist < MIN_DIST_ARANGE

class CourseJudgeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CourseJudgeWrapper, self).__init__(env)
        self._arrived = False

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        info = self._judge_course(obs)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if not self._arrived:
            return done

        if self._check_home(obs):
            print('***episode terminal while scout arrived***')
            return True
        return done

    def _judge_course(self, obs):
        if not self._arrived:
            self._arrived = self._check_arrived(obs)
            return self._arrived
        else:
            return True

    def _check_arrived(self, obs):
        scout = self.env.unwrapped.scout()
        enemy_base = self.env.unwrapped.enemy_base()
        dist =  sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     enemy_base[0], enemy_base[1])
        return dist < MIN_DIST_ARANGE

    def _check_home(self, obs):
        scout = self.env.unwrapped.scout()
        home = self.env.unwrapped.owner_base()
        dist =  sm.calculate_distance(scout.float_attr.pos_x,
                                     scout.float_attr.pos_y,
                                     home[0], home[1])
        return dist < MIN_DIST_ARANGE

