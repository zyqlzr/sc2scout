import gym
import numpy as np
from gym.spaces import Box
from sc2scout.wrapper.feature.scout_vec_feature import ScoutVecFeature, \
ScoutSimpleFeature

class ZergScoutObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ZergScoutObsWrapper, self).__init__(env)
        self._init_obs_space()
        self._map_size = self.env.unwrapped.map_size()
        self._reverse = False

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        obs = self.observation(obs)
        print('obswrapper reverse={},map_size={}'.format(
                self._reverse, self._map_size))
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    '''
    [scout_x,
     scout_y,
     home_x,
     home_y,
     enemy_base_x,
     enemy_base_y
    ]
    '''
    def _init_obs_space(self):
        low = np.zeros(6)
        high = np.ones(6)
        self.observation_space = Box(low, high)

    def _observation(self, obs):
        scout = self.env.unwrapped.scout()
        scout_pos = self.pos_transfer(scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = self.env.unwrapped.owner_base()
        enemy_pos = self.env.unwrapped.enemy_base()
        home_pos = self.pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self.pos_transfer(enemy_pos[0], enemy_pos[1])
        return np.array([float(scout_pos[0]) / self._map_size[0],
                         float(scout_pos[1]) / self._map_size[1],
                         float(home_pos[0]) / self._map_size[0],
                         float(home_pos[1]) / self._map_size[1],
                         float(enemy_pos[0]) / self._map_size[0],
                         float(enemy_pos[1]) / self._map_size[1]])

    def pos_transfer(self, x, y):
        if not self._reverse:
            return (x, y)
        cx = self._map_size[0] / 2
        cy = self._map_size[1] / 2
        pos_x = 0.0
        pos_y = 0.0
        if x > cx:
            pos_x = cx - abs(x - cx)
        else:
            pos_x = cx + abs(x - cx)

        if y > cy:
            pos_y = cy - abs(y - cy)
        else:
            pos_y = cy + abs(y - cy)

        return (pos_x, pos_y)

    def judge_reverse(self):
        scout = self.env.unwrapped.scout()
        if scout.float_attr.pos_x < scout.float_attr.pos_y:
            return False
        else:
            return True

class ZergOnewayObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ZergOnewayObsWrapper, self).__init__(env)
        self._init_obs_space()
        self._map_size = self.env.unwrapped.map_size()
        self._extractor = ScoutSimpleFeature()

    def _init_obs_space(self):
        self.observation_space = self._extractor.obs_space()

    def _reset(self):
        obs = self.env._reset()
        self._extractor.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def _observation(self, obs):
        features = self._extractor.extract(self.env, obs)
        return np.array(features)

class ZergScoutRoundTripObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ZergScoutRoundTripObsWrapper, self).__init__(env)
        self._init_obs_space()
        self._map_size = self.env.unwrapped.map_size()
        self._extractor = ScoutVecFeature()
        self._curr_status = False

    def _reset(self):
        obs = self.env._reset()
        self._extractor.reset(self.env)
        self._curr_status = False
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        self._curr_status = other
        obs = self.observation(obs)
        return obs, rwd, done, other

    def _observation(self, obs):
        features = self._extractor.extract(self.env, obs)
        if self._curr_status:
            features.append(float(1))
        else:
            features.append(float(0))
        return np.array(features)

    def _init_obs_space(self):
        low = np.zeros(9)
        high = np.ones(9)
        self.observation_space = Box(low, high)

