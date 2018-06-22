import gym
from sc2scout.wrapper.explore_enemy import scout_reward as sr

class ZergScoutRwdWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapper, self).__init__(env)
        self._rewards = [sr.HomeReward(), 
                         sr.EnemyBaseReward(), 
                         sr.ViewEnemyReward(),
                         sr.MinDistReward()]

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

class ZergScoutRwdWrapperV1(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapperV1, self).__init__(env)
        self._forward_rewards = [sr.HomeReward(),
                         sr.EnemyBaseReward(),
                         sr.ViewEnemyReward(),
                         sr.MinDistReward()]

        self._backward_rewards = [sr.HomeReward(back=True),
                         sr.EnemyBaseReward(back=True),
                         sr.MinDistReward()]

    def _reset(self):
        obs = self.env._reset()
        for fr in self._forward_rewards:
            fr.reset(obs, self.env.unwrapped)
        for br in self._backward_rewards:
            br.reset(obs, self.env.unwrapped)

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        if other:
            for r in self._backward_rewards:
                r.compute_rwd(obs, rwd, self.env.unwrapped)
                new_rwd += r.rwd
                return obs, new_rwd, done, other
        else:
            for r in self._forward_rewards:
                r.compute_rwd(obs, rwd, self.env.unwrapped)
                new_rwd += r.rwd
                return obs, new_rwd, done, other

class ZergScoutRwdWrapperV2(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapperV2, self).__init__(env)
        self._rewards = [sr.HomeReward(negative=True),
                         sr.EnemyBaseReward(negative=True), 
                         sr.ViewEnemyReward(weight=10),
                         sr.EnemyBaseArrivedReward(weight=30)]

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

class ZergScoutRwdWrapperV3(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapperV3, self).__init__(env)
        self._forward_rewards = [sr.HomeReward(negative=True),
                         sr.EnemyBaseReward(negative=True),
                         sr.ViewEnemyReward(weight=10),
                         sr.EnemyBaseArrivedReward(weight=30)]

        self._backward_rewards = [sr.HomeReward(back=True, negative=True),
                         sr.EnemyBaseReward(back=True, negative=True),
                         sr.HomeArrivedReward(weight=30)]

    def _reset(self):
        obs = self.env._reset()
        for fr in self._forward_rewards:
            fr.reset(obs, self.env.unwrapped)
        for br in self._backward_rewards:
            br.reset(obs, self.env.unwrapped)

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        if other:
            for r in self._backward_rewards:
                r.compute_rwd(obs, rwd, self.env.unwrapped)
                new_rwd += r.rwd
        else:
            for r in self._forward_rewards:
                r.compute_rwd(obs, rwd, self.env.unwrapped)
                new_rwd += r.rwd
        return obs, new_rwd, done, other

class ZergScoutRwdWrapperV4(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapperV4, self).__init__(env)
        self._rewards = [sr.HomeReward(negative=True),
                         sr.EnemyBaseReward(negative=True), 
                         sr.ViewEnemyReward(weight=10),
                         sr.MinDistReward(negative=True),
                         sr.EnemyBaseArrivedReward(weight=30)]

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

class ZergScoutRwdWrapperV5(gym.Wrapper):
    def __init__(self, env):
        super(ZergScoutRwdWrapperV5, self).__init__(env)
        self._forward_rewards = [sr.HomeReward(negative=True),
                         sr.EnemyBaseReward(negative=True),
                         sr.ViewEnemyReward(weight=10),
                         sr.EnemyBaseArrivedReward(weight=30),
                         sr.MinDistReward(negative=True)]

        self._backward_rewards = [sr.HomeReward(back=True, negative=True),
                         sr.EnemyBaseReward(back=True, negative=True),
                         sr.HomeArrivedReward(weight=30),
                         sr.MinDistReward(negative=True)]

    def _reset(self):
        obs = self.env._reset()
        for fr in self._forward_rewards:
            fr.reset(obs, self.env.unwrapped)
        for br in self._backward_rewards:
            br.reset(obs, self.env.unwrapped)

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        if other:
            for r in self._backward_rewards:
                r.compute_rwd(obs, rwd, self.env.unwrapped)
                new_rwd += r.rwd
        else:
            for r in self._forward_rewards:
                r.compute_rwd(obs, rwd, self.env.unwrapped)
                new_rwd += r.rwd
        return obs, new_rwd, done, other

