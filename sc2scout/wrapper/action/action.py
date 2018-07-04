
class Action(object):
    def __init__(self, env):
        self.env = env

    def reset(self):
        raise NotImplementedError

    def act(self, action):
        raise NotImplementedError

    def reverse_act(self, action):
        raise NotImplementedError

