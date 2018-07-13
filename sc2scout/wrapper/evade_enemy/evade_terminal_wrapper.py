import gym

class EvadeTerminalWrapper(gym.Wrapper):
    def __init__(self, env, max_step=4000, min_distance_allowed=5):
        super(EvadeTerminalWrapper, self).__init__(env)
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
