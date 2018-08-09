import gym
from sc2scout.wrapper.reward import evade_img_reward as ir
from sc2scout.wrapper.reward import scout_reward as sr
from sc2scout.wrapper.util.trip_status import TripStatus, TripCourse

class TargetTripRwd(gym.Wrapper):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(TargetTripRwd, self).__init__(env)
        self._forward_rewards = None
        self._explore_rewards = None
        self._backward_rewards = None
        self._final_rewards = None
        self._status = None
        self._compress_width = compress_width
        self._range_width = range_width
        self._explore_step = explore_step

    def _assemble_reward(self):
        raise NotImplementedError

    def _reset(self):
        self._assemble_reward()
        obs = self.env._reset()
        for fr in self._forward_rewards:
            fr.reset(obs, self.env.unwrapped)
        for br in self._backward_rewards:
            br.reset(obs, self.env.unwrapped)
        for er in self._explore_rewards:
            er.reset(obs, self.env.unwrapped)
        for r in self._final_rewards:
            r.reset(obs, self.env.unwrapped)
        self._init_status()
        return obs

    def _init_status(self):
        map_size = self.env.unwrapped.map_size()
        x_per_unit = map_size[0] / self._compress_width
        y_per_unit = map_size[1] / self._compress_width
        home_pos = self.env.unwrapped.owner_base()
        enemy_pos = self.env.unwrapped.enemy_base()
        x_range = x_per_unit * self._range_width
        y_range = y_per_unit * self._range_width
        self._status = TripCourse(home_pos, enemy_pos, (x_range, y_range), self._explore_step)
        self._status.reset()

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        scout = self.env.unwrapped.scout()
        status = self._status.check_status((scout.float_attr.pos_x, scout.float_attr.pos_y))

        #print('status=', status.value, 'backword=', TripStatus.BACKWORD.value)
        new_rwd = 0
        if status.value == TripStatus.BACKWORD.value:
            for r in self._backward_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd
                #print('backward rwd_step=', r.rwd)

            for r in self._final_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd
            #print('backward_rwd=', new_rwd)
            return obs, new_rwd, done, other
        elif status.value == TripStatus.EXPLORE.value:
            for r in self._explore_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                #print('explore rwd_step=', r.rwd)
                new_rwd += r.rwd

            for r in self._final_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd
            #print('explore_rwd=', new_rwd)
            return obs, new_rwd, done, other
        elif status.value == TripStatus.FORWARD.value:
            for r in self._forward_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                #print('forward rwd_step=', r.rwd)
                new_rwd += r.rwd

            for r in self._final_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd
            #print('forward_rwd=', new_rwd)
            return obs, new_rwd, done, other
        elif status.value == TripStatus.TERMINAL.value:
            for r in self._final_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd
            return obs, new_rwd, done, other

class ExploreTargetRwdWrapper(TargetTripRwd):
    def __init__(self, env, compress_width, range_width, explore_step):
        super(ExploreTargetRwdWrapper, self).__init__(env, 
              compress_width, range_width, explore_step)

    def _assemble_reward(self):
        self._forward_rewards = [sr.HomeReward(negative=True),
                                 sr.EnemyBaseReward(negative=True),
                                 sr.MinDistReward(negative=True)]
        self._backward_rewards = [sr.HomeReward(negative=True, back=True),
                                  sr.EnemyBaseReward(negative=True, back=True),
                                  ir.EvadeUnderAttackRwd()]
        self._explore_rewards = [ir.EvadeUnderAttackRwd(),
                                 ir.EvadeInTargetRangeRwd(self._compress_width, self._range_width),
                                 sr.ViewEnemyReward(weight=20),
                                 ir.EvadeTargetScanRwd(self._compress_width, self._range_width)]
        self._final_rewards = [ir.EvadeFinalRwd()]

