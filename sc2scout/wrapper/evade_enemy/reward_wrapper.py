import gym
from sc2scout.wrapper.reward import scout_reward as sr

class ScoutEvadeRwd(gym.Wrapper):
    def __init__(self, env):
        super(ScoutEvadeRwd, self).__init__(env)
        self._rewards = None


    def _assemble_reward(self):
        raise NotImplementedError

    def _reset(self):
        self._assemble_reward()
        obs = self.env._reset()
        for r in self._rewards:
            r.reset(obs, self.env.unwrapped)

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        for r in self._rewards:
            r.compute_rwd(obs, rwd, done, self.env.unwrapped)
            new_rwd += r.rwd
        return obs, new_rwd, done, other


class ZergScoutRwdWrapper(ScoutEvadeRwd):
    def __init__(self, env):
        super(ScoutEvadeRwd, self).__init__(env)

    def _assemble_reward(self):
        self._rewards = [sr.EvadeTimeReward(weight=30000),
                         sr.EvadeSpaceReward(weight=1),
                         sr.EvadeHealthReward(weight=1)]



