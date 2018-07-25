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
flags.DEFINE_integer("max_step", 10000, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("random_seed", None, "Random_seed used in game_core.")

flags.DEFINE_string("train_log_dir", './log/evade_cnnlstm_log', "train log directory")
flags.DEFINE_integer("checkpoint_freq", 5000, "load saved model")
flags.DEFINE_string("wrapper", 'evade_v1', "the name of wrapper")
flags.DEFINE_enum("agent_race", 'Z', races.keys(), "Agent's race.")
flags.DEFINE_enum("oppo_race", 'Z', races.keys(), "Opponent's race.")

flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")

flags.DEFINE_integer("max_episode_rwd", None, 'the max sum reward of episode')
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_integer('param_concurrent', 2, 'the concurrent')
flags.DEFINE_float('param_lam', 0.95, 'the parameter lan')
flags.DEFINE_float('param_gamma', 0.99, 'the parameter gamma')
flags.DEFINE_float('param_lr', 2.5e-4, 'the parameter learning rate')
flags.DEFINE_float('param_cr', 0.1, 'the parameter cliprange')
flags.DEFINE_integer('param_tstep', 100000, 'the parameter totoal step')

flags.DEFINE_string("map", 'ScoutSimple64Dodge2', "Name of a map to use.")
flags.mark_flag_as_required("map")
flags.mark_flag_as_required("wrapper")

mean_rwd_gap = 5
last_done_step = 0


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

    logger.configure(dir=FLAGS.train_log_dir, format_strs=['log'])

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

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf.Session(config=config).__enter__()

    #flags.DEFINE_float('param_tstep', 100000, 'the parameter totoal step')
    param_lam = FLAGS.param_lam
    param_gamma = FLAGS.param_gamma
    param_concurrent = FLAGS.param_concurrent
    param_lr = FLAGS.param_lr
    param_cr = FLAGS.param_cr
    param_tstep = FLAGS.param_tstep
    print('params, lam={} gamma={} concurrent={} lr={} tstep={}'.format(
        param_lam, param_gamma, param_concurrent, param_lr, param_tstep))

    env = make_sc2_dis_env(num_env=param_concurrent, seed=rs, players=players, agent_interface_format=agent_interface_format)

    ppo2.learn(policy=CnnPolicy, env=env, nsteps=128, nminibatches=1,
               lam=param_lam, gamma=param_gamma, noptepochs=4, log_interval=1,
               ent_coef=0.01,
               lr=lambda f: f * param_lr,
               cliprange=lambda f: f * param_cr,
               total_timesteps=param_tstep,
               save_interval=10)

def entry_point():  # Needed so setup.py scripts work.
    app.run(main)

if __name__ == "__main__":
    app.run(main)
