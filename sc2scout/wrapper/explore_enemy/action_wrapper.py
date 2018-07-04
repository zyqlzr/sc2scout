import gym
import numpy as np
from sc2scout.wrapper.action.scout_action import ScoutAction

class ZergScoutActWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ZergScoutActWrapper, self).__init__(env)
        self._reverse = False
        self._act = None

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        print('action wrapper, reverse={}', self._reverse)
        self._act = ScoutAction(self.env, self._reverse)
        return obs

    def _step(self, action):
        action = self._action(action)
        return self.env._step(action)

    def _action(self, action):
        return self._act.act(action)

    def _reverse_action(self, action):
        raise NotImplementedError()

    def judge_reverse(self):
        scout = self.env.unwrapped.scout()
        if scout.float_attr.pos_x < scout.float_attr.pos_y:
            return False
        else:
            return True

