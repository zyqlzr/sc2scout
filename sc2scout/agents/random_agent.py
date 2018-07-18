import gym
from gym import wrappers, logger

from sc2scout.agents.agent_base import AgentBase

class RandomAgent(AgentBase):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def reset(self, env):
        pass
