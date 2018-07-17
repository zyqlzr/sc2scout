
class AgentBase(object):
    def act(self, observation, reward, done):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
