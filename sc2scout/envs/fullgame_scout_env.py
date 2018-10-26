import gym
import numpy as np

from pysc2.env import sc2_env
from pysc2.env.environment import StepType

from tstarbot.agents.zerg_agent import ZergAgent
import tstarbot.data.pool.scout_pool as sp
from sc2scout.envs import scout_macro as sm

FULLGAME_MAP_SIZE = {
    'Simple64': (88, 96),
    'ScoutSimple64': (88, 96),
    'AbyssalReef': (200, 176),
    'ScoutAbyssalReef': (200, 176),
    'Acolyte': (168, 200),
}

MAX_STEP_DEF = 10000

class FullGameScoutEnv(gym.Env):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        if self._kwargs.get('config_path'):  # use the config file
            config_path = kwargs['config_path']
            self._full_agent = ZergAgent(config_path=config_path)
            self._kwargs.remove('config_path')
        else:
            self._full_agent = ZergAgent()

        self._sc2env = None
        self._episode = 0
        self._num_step = 0
        self._max_step_num = MAX_STEP_DEF
        self._episode_reward = 0
        self._total_reward = 0
        self._last_obs = None
        self._last_actions = None

        self._fullgame_pre_step = False
        self._scout = None
        self._owner_base_pos = None
        self._map_size = None

        #map{id, pos}
        self._target_bases = {}

        self._init_action_space()
        self._init_map_size(self._kwargs['map_name'])

    def _init_action_space(self):
        self.action_space = gym.spaces.Discrete(8)

    def _init_map_size(self, map_name):
        self._map_size = FULLGAME_MAP_SIZE[map_name]
        print('fullgame map_name={},MapSize={}'.format(map_name, self._map_size))

    def _reset(self):
        if self._sc2env is None:
            self._sc2env = sc2_env.SC2Env(**self._kwargs)
        if self._episode > 0:
            print("---Episode {} ended with reward {} after {} steps.---".format(
                        self._episode, self._episode_reward, self._num_step))
            print("---Got {} total reward so far, with an average reward of {} per episode---".format(
                        self._total_reward, float(self._total_reward) / self._episode))
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        self._fullgame_pre_step = False
        print("Episode %d starting... ", self._episode)

        self._full_agent.reset()
        self._last_obs = self._sc2env.reset()
        self._reset_dc()
        self._init_target_and_scout()
        return self._last_obs[0]

    def _step(self, scout_action):
        self._num_step += 1
        self._push_scout_action_to_am(scout_action)
        self._last_actions = [self._full_agent.step(timestep) for timestep in self._last_obs]
        #print("fullgame agent actions=", self._last_actions)

        try:
            self._last_obs = self._sc2env.step(self._last_actions)
        except KeyboardInterrupt:
            print("Interrupted. Quitting...")
            return None, 0, True, {}
        except Exception:
            print("exception while execute action")
            return None, 0, True, {}
        reward = self._last_obs[0].reward
        self._episode_reward += reward
        self._total_reward += reward
        return self._last_obs[0], reward, self._last_obs[0].step_type == StepType.LAST, {}

    def _close(self):
        if self._episode > 0:
            print("Episode %d ended with reward %d after %d steps.",
                        self._episode, self._episode_reward, self._num_step)
            print("Got %d total reward, with an average reward of %g per episode",
                        self._total_reward, float(self._total_reward) / self._episode)
        if self._sc2env is not None:
            self._sc2env.close()
        super()._close()

    def _reset_dc(self):
        dc = self._full_agent.dc
        for ts in self._last_obs:
            dc.update(ts)

    def _init_target_and_scout(self):
        spool = self._full_agent.dc.dd.scout_pool
        self._scout = spool.select_scout()
        self._scout.is_doing_task = True
        self._owner_base_pos = spool.home_pos

        #print("scout_pool home_pos={},target_num={},scout_num={}".format(
        #      spool.home_pos, spool.scout_base_target_num(),
        #      len(spool.list_scout())))
        #print("fullgame scout={},target={}".format(self._scout, self._target))

        #self._target = spool.find_enemy_subbase_target()
        #self._target.has_scout = True
        self._init_target_bases()
        self._target_id = self._find_furthest_target()
        self._fullgame_scout_flag = True
        print("fullgame home={},target={}".format(self._owner_base_pos,
                                                  self._target_bases[self._target_id]))

    def _dist(self, base):
        pos = base.pos
        dist = sm.calculate_distance(self._owner_base_pos[0],
                                     self._owner_base_pos[1],
                                     pos[0], pos[1])
        return dist

    def _init_target_bases(self):
        spool = self._full_agent.dc.dd.scout_pool
        bases = spool.get_scout_targets()
        old_bases = bases
        bases.sort(key=lambda x: self._dist(x))
        for i in range(len(bases)):
            self._target_bases[i] = bases[i].pos
        print('base_positions=', self._target_bases)

    def _find_furthest_target(self):
        furthest_dist = 0.0
        furthest_base = None
        for key, value in self._target_bases.items():
            dist = sm.calculate_distance(self._owner_base_pos[0],
                                         self._owner_base_pos[1],
                                         value[0], value[1])
            if furthest_dist < dist:
                furthest_dist = dist
                furthest_base = key
        return furthest_base

    def _push_scout_action_to_am(self, action):
        self._full_agent.am.push_actions(action[0])

    def scout(self):
        return self._scout.unit()

    def scout_survive(self):
        return not self._scout.is_lost()

    def owner_base(self):
        return self._owner_base_pos

    def enemy_base(self):
        return self._target_bases[self._target_id]

    def map_size(self):
        return self._map_size

    def step_number(self):
        return self._num_step

    def max_step_number(self):
        return self._max_step_num

    def save_replay(self, replay_dir):
        pass

    def judge_reverse(self):
        home = self._owner_base_pos
        if home[0] < home[1]:
            return False
        else:
            return True

    def get_id_by_pos(self, pos):
        min_dist = None
        min_base_gap = 5
        base_id = None
        for key, value in self._target_bases.items():
            dist = sm.calculate_distance(pos[0], pos[1], value[0], value[1])
            if min_dist is None:
                min_dist = dist
                base_id = key
            elif min_dist > dist:
                min_dist = dist
                base_id = key

        if min_dist > min_base_gap:
            return None
        else:
            return base_id

