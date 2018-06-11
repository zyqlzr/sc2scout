import logging

import gym
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SC2GymEnv(gym.Env):
    metadata = {'render.modes': [None, 'human'],
                'action.noop': 8}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def _step(self, action):
        self._num_step += 1
        #print('env action=', action)
        try:
            obs = self._env.step(action)[0]
        except KeyboardInterrupt:
            logger.info("Interrupted. Quitting...")
            return None, 0, True, {}
        except Exception:
            logger.exception("exception while execute action")
            return None, 0, True, {}
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def _reset(self):
        if self._env is None:
            self._init_env()
        if self._episode > 0:
            logger.info("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            logger.info("Got %d total reward so far, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        logger.info("Episode %d starting...", self._episode)
        obs = self._env.reset()[0]
        return obs

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
