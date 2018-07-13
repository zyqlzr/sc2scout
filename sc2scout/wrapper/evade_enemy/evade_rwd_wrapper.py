import gym
from sc2scout.wrapper.reward import evade_reward as er

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


class ScoutEvadeRwdWrapper(ScoutEvadeRwd):
    def __init__(self, env):
        super(ScoutEvadeRwdWrapper, self).__init__(env)

    def _assemble_reward(self):
        self._rewards = [er.EvadeTimeReward(weight=30000),
                         er.EvadeSpaceReward(weight=1),
                         er.EvadeHealthReward(weight=1)]



