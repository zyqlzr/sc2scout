import logging

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

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._env = None

        self._obs_oppo = None

        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0

    def _step(self, action):
        self._num_step += 1

        try:
            action_oppo = self._step_oppo()

            actions = []

            if action is not None:
                actions += action

            if action_oppo is not None:
                actions += action_oppo

            timesteps = self._env.step(actions)
            obs = timesteps[0]
            self._obs_oppo = timesteps[1]
        except KeyboardInterrupt:
            logger.info("Interrupted. Quitting...")
            return None, 0, True, {}
        #except Exception:
        #    logger.exception("exception while execute action")
        #    return None, 0, True, {}
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return obs, reward, obs.step_type == StepType.LAST, {}

    def _step_oppo(self):
        # obs: self._obs_oppo
        # return [actions]

        zergling = None
        enemy = None

        for u in self._obs_oppo.observation['units']:
            if u.int_attr.unit_type == UNIT_TYPEID.ZERG_ZERGLING.value:
                zergling = u
            elif u.int_attr.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value:
                enemy = u

        if zergling is None or enemy is None:
            return None

        #print([enemy.float_attr.pos_x, enemy.float_attr.pos_y])
        return [[self._move_to_target(zergling, [enemy.float_attr.pos_x,
                                            enemy.float_attr.pos_y])]]

    def _move_to_target(self, u, pos):
        action = sc_pb.Action()
        action.action_raw.unit_command.ability_id = ABILITY_ID.SMART.value
        action.action_raw.unit_command.target_world_space_pos.x = pos[0]
        action.action_raw.unit_command.target_world_space_pos.y = pos[1]
        action.action_raw.unit_command.unit_tags.append(u.tag)
        return action

    def _reset(self):
        if self._env is None:
            self._init_env()
        if self._episode > 0:
            #logger.info("---Episode %d ended with reward %d after %d steps.---",
            #            self._episode, self._episode_reward, self._num_step)
            #logger.info("---Got %d total reward so far, with an average reward of %g per episode---",
            #            self._total_reward, float(self._total_reward) / self._episode)
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
        self._obs_oppo = timesteps[1]
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
