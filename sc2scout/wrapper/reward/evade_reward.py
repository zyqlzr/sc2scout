from sc2scout.wrapper.reward.reward import Reward
import math

class EvadeTimeReward(Reward):
    def __init__(self, weight=1):
        super(EvadeTimeReward, self).__init__(weight)

    def reset(self, obs, env):
        self._curr_step = 0

    def compute_rwd(self, obs, reward, done, env):
        curr_step = env.curr_step()
        game_step_per_episode = env.episode_length()
        normorlized_curr_step = float(curr_step)/game_step_per_episode

        self.rwd = 1 - math.pow((1-normorlized_curr_step),0.4)
        self.rwd = self.w * self.rwd
        print('EvadeTimeReward reward=', self.rwd)

class EvadeSpaceReward(Reward):
    def __init__(self, weight=1):
        super(EvadeSpaceReward, self).__init__(weight)


    def reset(self, obs, env):
        self._map_size = env.map_size()

    def compute_rwd(self, obs, reward, done, env):
        viewEnemy = env.view_enemy()

        enemy = env.enemy()
        if not viewEnemy:
            self.rwd = 0
        else:
            if enemy:
                scout_x = env.scout().float_attr.pos_x/float(self._map_size[0])
                scout_y = env.scout().float_attr.pos_y/float(self._map_size[1])
                enemy_x = env.enemy().float_attr.pos_x/float(self._map_size[0])
                enemy_y = env.enemy().float_attr.pos_y/float(self._map_size[1])
                curr_dist = env._calculate_distances(scout_x,scout_y,enemy_x,enemy_y)

                self.rwd = math.pow(curr_dist,0.4) - 1
        self.rwd = self.dynamicWeight(env) * self.rwd

        print('EvadeSpaceReward reward=', self.rwd)

    def dynamicWeight(self,env):
        curr_step = env.curr_step()
        game_step_per_episode = env.episode_length()
        return self.w * math.pow(float(curr_step)/game_step_per_episode,0.4)*10000

class EvadeHealthReward(Reward):
    def __init__(self, weight=1):
        super(EvadeHealthReward, self).__init__(weight)

    def reset(self, obs, env):
        self.scout_max_health = env.scout().float_attr.health_max

    def compute_rwd(self, obs, reward, done, env):
        normorlized_health = env.scout().float_attr.health/float(self.scout_max_health)
        self.rwd = math.pow(normorlized_health, 0.4) - 1
        self.rwd = self.dynamicWeight(env) * self.rwd

        print('EvadeHealthReward reward=', self.rwd)


    def dynamicWeight(self,env):
        curr_step = env.curr_step()
        game_step_per_episode = env.episode_length()
        return self.w * math.pow(float(curr_step)/game_step_per_episode,0.4)*10000

