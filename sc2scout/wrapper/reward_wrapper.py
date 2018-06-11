import gym
from sc2scout.wrapper import scout_reward as sr

class ZergScoutRwdWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapper, self).__init__(env)
        self._rewards = [sr.HomeReward(), 
                         sr.EnemyBaseReward(), 
                         sr.ViewEnemyReward()]

    def _reset(self):
        obs = self.env._reset()
        for r in self._rewards:
            r.reset(obs, self.env.unwrapped)

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        for r in self._rewards:
            r.compute_rwd(obs, rwd, self.env.unwrapped)
            new_rwd += r.rwd
        return obs, new_rwd, done, other

