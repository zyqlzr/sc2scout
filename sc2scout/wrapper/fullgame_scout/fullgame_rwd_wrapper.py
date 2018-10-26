import gym
from sc2scout.wrapper.reward import evade_img_reward as ir
from sc2scout.wrapper.reward import fullgame_reward as fr
from sc2scout.wrapper.util.dest_range import DestRange

class FullGameSimpleRwd(gym.Wrapper):
    def __init__(self, env):
        super(FullGameSimpleRwd, self).__init__(env)
        self._rewards = None

    def _assemble_reward(self):
        raise NotImplementedError

    def _reset(self):
        obs = self.env._reset()
        self._assemble_reward()
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


class FullGameRoundTripRwd(gym.Wrapper):
    def __init__(self, env):
        super(FullGameRoundTripRwd, self).__init__(env)
        self._first_rewards = None
        self._second_rewards = None
        self._final_rewards = None

    def _assemble_reward(self):
        raise NotImplementedError

    def _condition_judge(self, env):
        raise NotImplementedError

    def _reset(self):
        obs = self.env._reset()
        self._assemble_reward()
        for fr in self._first_rewards:
            fr.reset(obs, self.env.unwrapped)
        for br in self._second_rewards:
            br.reset(obs, self.env.unwrapped)
        for r in self._final_rewards:
            r.reset(obs, self.env.unwrapped)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        status = self._condition_judge(self.env)
        new_rwd = 0
        if status:
            for r in self._second_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd

            for r in self._final_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd
            return obs, new_rwd, done, other
        else:
            for r in self._first_rewards:
                r.compute_rwd(obs, rwd, done, self.env.unwrapped)
                new_rwd += r.rwd

            for r in self._final_rewards:
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


class FullGameRwdWrapperV2(FullGameRoundTripRwd):
    def __init__(self, env, target_range):
        super(FullGameRwdWrapperV2, self).__init__(env)
        self._target_range = target_range
        self._target = None

    def _assemble_reward(self):
        self._target = DestRange(self.env.unwrapped.enemy_base(), 
                                 dest_range=self._target_range)

        self._first_rewards = [fr.FullGameMoveToTarget(self._target_range)]
        self._second_rewards = [ir.EvadeUnderAttackRwd(),
                                fr.FullGameInTargetRangeRwd(self._target_range),
                                fr.FullGameViewRwd(weight=5)]
        self._final_rewards = [ir.EvadeFinalRwd()]

    def _condition_judge(self, env):
        scout = env.unwrapped.scout()
        scout_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        return self._target.in_range(scout_pos)


class FullGameRwdWrapperV3(FullGameRoundTripRwd):
    def __init__(self, env, target_range):
        super(FullGameRwdWrapperV3, self).__init__(env)
        self._target_range = target_range
        self._target = None

    def _assemble_reward(self):
        self._target = DestRange(self.env.unwrapped.enemy_base(), 
                                 dest_range=self._target_range)

        self._first_rewards = [fr.FullGameMoveToTarget(self._target_range)]
        self._second_rewards = [ir.EvadeUnderAttackRwd(),
                                fr.FullGameInTargetRangeRwd(self._target_range),
                                fr.FullGameViewRwdV1()]
        self._final_rewards = [fr.FullGameFinalRwd()]

    def _condition_judge(self, env):
        scout = env.unwrapped.scout()
        scout_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        return self._target.in_range(scout_pos)

