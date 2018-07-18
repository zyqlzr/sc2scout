import gym
import numpy as np
from sc2scout.wrapper.action.scout_action import ScoutAction

class EvadeActWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(EvadeActWrapper, self).__init__(env)
        self._reverse = False
        self._act = None

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        self._act = ScoutAction(self.env, self._reverse, move_range=0.5)
        print('evade action,reverse={},move_range=0.2', self._reverse)
        return obs

    def _step(self, action):
        action = self._action(action)
        return self.env._step(action)

    def _action(self, action):
        return self._act.act(action)

    def _reverse_action(self, action):
        raise NotImplementedError()

    def judge_reverse(self):
        home = self.env.unwrapped.owner_base()
        if home[0] < home[1]:
            return False
        else:
            return True


