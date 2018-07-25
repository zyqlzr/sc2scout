from baselines import deepq
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import multiprocessing
import tensorflow as tf
import sys

from pysc2 import maps
from pysc2.env import sc2_env
from sc2scout.envs import ZergScoutSelfplayEnv
from sc2scout.wrapper.wrapper_factory import make, model
from sc2scout.wrapper.util.sc2_params import races
from sc2scout.agents import ZergBotAgent

from absl import app
from absl import flags
import time
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")

flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_float("screen_ratio", "1.33",
                   "Screen ratio of width / height")
flags.DEFINE_string("agent_interface_format", "feature",
                    "Agent Interface Format: [feature|rgb]")

flags.DEFINE_integer("max_agent_episodes", 1, "Total agent episodes.")
flags.DEFINE_integer("max_step", 4000, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("random_seed", None, "Random_seed used in game_core.")

flags.DEFINE_string("model_dir", './log/evade_cnnlstm_log/checkpoints/00001', "model directory")
flags.DEFINE_string("wrapper", 'evade_v1', "the name of wrapper")
flags.DEFINE_enum("agent_race", 'Z', races.keys(), "Agent's race.")
flags.DEFINE_enum("oppo_race", 'Z', races.keys(), "Opponent's race.")

flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")

flags.DEFINE_integer("max_episode_rwd", None, 'the max sum reward of episode')
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_float('param_lr', 1e-4, 'learning rate')
flags.DEFINE_integer('param_bf', 50000, 'buffer size')
flags.DEFINE_float('param_ef', 0.1, 'explore_fraction')
flags.DEFINE_float('param_efps', 0.02, 'exploration_final_eps')

flags.DEFINE_string("map", 'ScoutSimple64Dodge2', "Name of a map to use.")
flags.mark_flag_as_required("map")
flags.mark_flag_as_required("wrapper")

mean_rwd_gap = 5
last_done_step = 0

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    step = lcl['t']
    reset_flag = lcl['reset']

    if reset_flag:
        global last_done_step
        mr = lcl['mean_100ep_reward']
        rwds = lcl['episode_rewards'][-2:-1]
        logger.log('last_done_step={} step={} last_episode_rwd={}, mean_100ep_rwd={}'.format(
                last_done_step, step, rwds, mr))
        last_done_step = step
        if FLAGS.max_episode_rwd is None or len(rwds) == 0:
            return False

        if mr <= (FLAGS.max_episode_rwd - mean_rwd_gap):
            return False

        if rwds[0] >= FLAGS.max_episode_rwd:
            logger.log('episode reward get the max_reward, stop and save model, {} {}'.format(
                    FLAGS.max_episode_rwd, rwds[0]))
            return True
    return False

def make_sc2_dis_env(num_env, seed,players,agent_interface_format,start_index=0):
    """
    Create a wrapped SubprocVecEnv for SCII.
    """
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            agents = [ZergBotAgent()]

            env = ZergScoutSelfplayEnv(
                agents,
                map_name=FLAGS.map,
                players=players,
                step_mul=FLAGS.step_mul,
                random_seed=seed,
                game_steps_per_episode=FLAGS.max_step,
                agent_interface_format=agent_interface_format,
                score_index=-1,  # this indicates the outcome is reward
                disable_fog=FLAGS.disable_fog,
                visualize=FLAGS.render
            )

            env = make(FLAGS.wrapper, env)
            return env
        return _thunk
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main(unused_argv):
    rs = FLAGS.random_seed
    if FLAGS.random_seed is None:
        rs = int((time.time() % 1) * 1000000)

    players = []
    players.append(sc2_env.Agent(races[FLAGS.agent_race]))
    players.append(sc2_env.Agent(races[FLAGS.oppo_race]))

    screen_res = (int(FLAGS.screen_ratio * FLAGS.screen_resolution) // 4 * 4,
                  FLAGS.screen_resolution)
    if FLAGS.agent_interface_format == 'feature':
        agent_interface_format = sc2_env.AgentInterfaceFormat(
        feature_dimensions = sc2_env.Dimensions(screen=screen_res,
            minimap=FLAGS.minimap_resolution))
    elif FLAGS.agent_interface_format == 'rgb':
        agent_interface_format = sc2_env.AgentInterfaceFormat(
        rgb_dimensions=sc2_env.Dimensions(screen=screen_res,
            minimap=FLAGS.minimap_resolution))
    else:
        raise NotImplementedError

    agents = [ZergBotAgent()]

    ncpu = 1
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf.Session(config=config).__enter__()

    # env = make_sc2_dis_env(num_env=1, seed=rs, players=players, agent_interface_format=agent_interface_format)
    model_dir = FLAGS.model_dir

    total_rwd = 0

    env = ZergScoutSelfplayEnv(
        agents,
        map_name=FLAGS.map,
        players=players,
        step_mul=FLAGS.step_mul,
        random_seed=rs,
        game_steps_per_episode=FLAGS.max_step,
        agent_interface_format=agent_interface_format,
        score_index=-1,  # this indicates the outcome is reward
        disable_fog=FLAGS.disable_fog,
        visualize=FLAGS.render
    )

    env = make(FLAGS.wrapper, env)
    agent = ppo2.load_model(CnnPolicy,env,model_dir)

    try:
        obs = env.reset()
        state = agent.initial_state
        n_step = 0
        done = False

        # run this episode
        while True:
            n_step += 1
            obs = np.reshape(obs, (1,) + obs.shape)  # convert shape (32,32,20) to (1,32,32,20)
            action, value, state, _ = agent.step(obs, state, done)
            obs, rwd, done, info = env.step(action)
            print('action=', action, '; rwd=', rwd)
            # print('step rwd=', rwd, ',action=', action, "obs=", obs)
            total_rwd += rwd
            if done:
                print("game over, total_rwd=", total_rwd)
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("evaluation over")
    env.unwrapped.save_replay('evaluate')
    # env.close()


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)

if __name__ == "__main__":
    app.run(main)
