import gym

class FullGameTerminalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FullGameTerminalWrapper, self).__init__(env)

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        obs, rwd, done, info = self.env._step(action)
        return obs, rwd, self._check_terminal(obs, done), info

    def _check_terminal(self, obs, done):
        if done:
            return done

        survive = self.env.unwrapped.scout_survive()
        if not survive:
            return True

        return done

