import gym
import numpy as np
from sc2scout.wrapper.action.scout_action import ScoutAction

class FullGameActWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(FullGameActWrapper, self).__init__(env)
        self._reverse = False
        self._act = None

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.env.unwrapped.judge_reverse()
        self._act = ScoutAction(self.env, self._reverse, move_range=0.5)
        print('fullgame action,reverse={}, move_range=0.2', self._reverse)
        return obs

    def _step(self, action):
        action = self._action(action)
        return self.env._step(action)

    def _action(self, action):
        return self._act.act(action)

    def _reverse_action(self, action):
        raise NotImplementedError()

