import gym
from sc2scout.envs import SC2GymEnv

class ZergScoutWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutWrapper, self).__init__(env)

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        return self.env._step(action)
