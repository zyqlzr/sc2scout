import gym
from sc2scout.wrapper.reward import evade_img_reward as ir
from sc2scout.wrapper.reward import fullgame_reward as fr

class FullGameSimpleRwd(gym.Wrapper):
    def __init__(self, env):
        super(FullGameSimpleRwd, self).__init__(env)
        self._rewards = None

    def _assemble_reward(self):
        raise NotImplementedError

    def _reset(self):
        self._assemble_reward()
        obs = self.env._reset()
        for r in self._rewards:
            r.reset(obs, self.env.unwrapped)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        for r in self._rewards:
            r.compute_rwd(obs, rwd, done, self.env.unwrapped)
            new_rwd += r.rwd
        return obs, new_rwd, done, other


class FullGameRwdWrapper(FullGameSimpleRwd):
    def __init__(self, env, compress_width, range_width):
        super(FullGameRwdWrapper, self).__init__(env)
        self._compress_width = compress_width
        self._range_width = range_width

    def _assemble_reward(self):
        self._rewards = [ir.EvadeUnderAttackRwd(),
                         ir.EvadeInTargetRangeRwd(self._compress_width, self._range_width),
                         fr.FullGameViewRwd(weight=5),
                         ir.EvadeFinalRwd()]


class FullGameRwdWrapperV1(FullGameSimpleRwd):
    def __init__(self, env, target_range):
        super(FullGameRwdWrapperV1, self).__init__(env)
        self._target_range = target_range

    def _assemble_reward(self):
        self._rewards = [ir.EvadeUnderAttackRwd(),
                         fr.FullGameInTargetRangeRwd(self._target_range),
                         fr.FullGameViewRwd(weight=5),
                         ir.EvadeFinalRwd()]


