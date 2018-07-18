import logging
import random

import gym
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions
from pysc2.lib.typeenums import UNIT_TYPEID, ABILITY_ID
from s2clientprotocol import sc2api_pb2 as sc_pb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SC2SelfplayGymEnv(gym.Env):
    metadata = {'render.modes': [None, 'human'],
                'action.noop': 8}

    def __init__(self, agents, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None
        if not isinstance(agents, list) and not isinstance(agents, tuple):
            raise Exception('input agents is invalid')
        self._agents = agents
        self._curr_agent_index = 0

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def _step(self, action):
        self._num_step += 1

        try:
            agent_action = self._agent_step()
            #print('action=', action[0], ';agent_action=', agent_action)
            actions = []
            actions += action
            actions += agent_action
            #print('actions= ', actions)

            timesteps = self._env.step(actions)
            obs = timesteps[0]
            self._agent_obs = timesteps[1]
        except KeyboardInterrupt:
            logger.info("Interrupted. Quitting...")
            return None, 0, True, {}

        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def _agent_step(self):
        reward = self._agent_obs.reward
        done = self._agent_obs.step_type == StepType.LAST
        return [self._agents[self._curr_agent_index].act(self._agent_obs, reward, done)]

    def _reset(self):
        if self._env is None:
            self._init_env()
        if self._episode > 0:
            print("---Episode {} ended with reward {} after {} steps.---".format(
                        self._episode, self._episode_reward, self._num_step))
            print("---Got {} total reward so far, with an average reward of {} per episode---".format(
                        self._total_reward, float(self._total_reward) / self._episode))
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0

        logger.info("Episode %d starting...", self._episode)
        timesteps = self._env.reset()
        obs = timesteps[0]
        self._agent_obs = timesteps[1]
        return obs

    def _select_agent(self):
        if len(self._agents) == 1:
            self._curr_agent_index = 0

        self._curr_agent_index = random.randint(0, len(self._agents))

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        print("env params:", self._kwargs)
        self._env = sc2_env.SC2Env(**self._kwargs)

    def _close(self):
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
        if self._env is not None:
            self._env.close()

        super()._close()

    @property
    def settings(self):
        return self._kwargs

    @property
    def action_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.action_spec()

    @property
    def observation_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward
