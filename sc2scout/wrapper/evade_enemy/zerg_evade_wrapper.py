import gym

class ZergEvadeWrapper(gym.Wrapper):
    def __init__(self, env):
        print('ZergEvadeWrapper initilize')
        super(ZergEvadeWrapper, self).__init__(env)
        print('act_space_shape=', self.action_space.shape)
        print('obs_space_shape=', self.observation_space.shape)

    def _reset(self):
        return self.env._reset()

    def _step(self, action):
        return self.env._step(action)
