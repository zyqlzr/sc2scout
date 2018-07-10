from baselines import deepq
from baselines import logger

from pysc2 import maps
from pysc2.env import sc2_env
from sc2scout.envs import ZergScoutEnv,ZergEvadeEnv
from sc2scout.wrapper.wrapper_factory import make, model

from absl import app
from absl import flags
import time

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_episodes", 1, "Total agent episodes.")
flags.DEFINE_integer("max_step", 10000, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
# init 8 since sc2_env initilization period will class _step() function once, each time 8*4(1+3_skip) steps taken
flags.DEFINE_integer("random_seed", None, "Random_seed used in game_core.")

flags.DEFINE_string("train_log_dir", './log/evade_log', "train log directory")
flags.DEFINE_string("checkpoint_path", './model_save/evade_model/model6/', "load saved model")
flags.DEFINE_integer("checkpoint_freq", 5000, "load saved model")
flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run")
flags.DEFINE_string("agent_config", "",
                    "Agent's config in py file. Pass it as python module."
                    "E.g., tstarbot.agents.dft_config")
flags.DEFINE_string("wrapper", 'evade_v0', "the name of wrapper")
flags.DEFINE_enum("agent_race", 'Z', sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", 'Z', sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 2, "How many instances to run in parallel.")
flags.DEFINE_integer("max_episode_rwd", None, 'the max sum reward of episode')
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_float('param_lr', 1e-4, 'learning rate')
flags.DEFINE_integer('param_bf', 50000, 'buffer size')
flags.DEFINE_float('param_ef', 0.1, 'explore_fraction')
flags.DEFINE_float('param_efps', 0.02, 'exploration_final_eps')

flags.DEFINE_string("map", 'scout_evade', "Name of a map to use.")
flags.mark_flag_as_required("map")
flags.mark_flag_as_required("wrapper")

mean_rwd_gap = 5
last_done_step = 0 

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    step = lcl['t']
    reset_flag = lcl['reset']
    '''
    if reset_flag:
        global last_done_step
        logger.log('last_done_step={} step={} last_episode_rwd={}'.format(
                last_done_step, step, lcl['episode_rewards'][-2:-1]))
        last_done_step = step
    '''

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

def main(unused_argv):
    #env = gym.make("SC2GYMENV-v0")
    #env.settings['map_name'] = 'ScoutSimple64'

    rs = FLAGS.random_seed
    if FLAGS.random_seed is None:
        rs = int((time.time() % 1) * 1000000)

    logger.configure(dir=FLAGS.train_log_dir, format_strs=['log'])

    # env = ZergScoutEnv(
    #         map_name=FLAGS.map,
    #         agent_race=FLAGS.agent_race,
    #         bot_race=FLAGS.bot_race,
    #         difficulty=FLAGS.difficulty,
    #         step_mul=FLAGS.step_mul,
    #         random_seed=rs,
    #         game_steps_per_episode=FLAGS.max_step,
    #         screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
    #         minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
    #         score_index=-1,  # this indicates the outcome is reward
    #         disable_fog=FLAGS.disable_fog,
    #         visualize=FLAGS.render
    #     )


    env = ZergEvadeEnv(
        map_name=FLAGS.map,
        agent_race=FLAGS.agent_race,
        bot_race=FLAGS.bot_race,
        difficulty=FLAGS.difficulty,
        step_mul=FLAGS.step_mul,
        random_seed=rs,
        game_steps_per_episode=FLAGS.max_step,
        screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
        minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
        score_index=-1,  # this indicates the outcome is reward
        disable_fog=FLAGS.disable_fog,
        visualize=FLAGS.render
    )

    env = make(FLAGS.wrapper, env)

    network = model(FLAGS.wrapper) #deepq.models.mlp([64, 32])

    print('params, lr={} bf={} ef={} ef_eps={}'.format(
            FLAGS.param_lr, FLAGS.param_bf, FLAGS.param_ef, FLAGS.param_efps))

    act = deepq.learn(
        env,
        q_func=network,
        lr=FLAGS.param_lr,
        max_timesteps=100000,
        buffer_size=FLAGS.param_bf,
        exploration_fraction=FLAGS.param_ef,
        exploration_final_eps=FLAGS.param_efps,
        checkpoint_path=FLAGS.checkpoint_path,
        checkpoint_freq=FLAGS.checkpoint_freq,
        print_freq=10,
        callback=callback
    )

def entry_point():  # Needed so setup.py scripts work.
    app.run(main)

if __name__ == "__main__":
    app.run(main)
