from baselines import deepq
from baselines import logger

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
flags.DEFINE_integer("max_step", 4000, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("random_seed", None, "Random_seed used in game_core.")

flags.DEFINE_string("train_log_dir", './log', "train log directory")
flags.DEFINE_string("checkpoint_path", './model_save', "load saved model")
flags.DEFINE_integer("checkpoint_freq", 5000, "load saved model")
flags.DEFINE_string("wrapper", None, "the name of wrapper")
flags.DEFINE_enum("agent_race", 'Z', races.keys(), "Agent's race.")
flags.DEFINE_enum("oppo_race", 'Z', races.keys(), "Opponent's race.")

flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")

flags.DEFINE_integer("max_episode_rwd", None, 'the max sum reward of episode')
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_float('param_lr', 1e-4, 'learning rate')
flags.DEFINE_integer('param_bf', 50000, 'buffer size')
flags.DEFINE_float('param_ef', 0.1, 'explore_fraction')
flags.DEFINE_float('param_efps', 0.02, 'exploration_final_eps')

flags.DEFINE_string("map", None, "Name of a map to use.")
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

    agents = [ZergBotAgent()]

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

    network = model(FLAGS.wrapper) #deepq.models.mlp([64, 32])

    print('params, lr={} bf={} ef={} ef_eps={}'.format(
            FLAGS.param_lr, FLAGS.param_bf, FLAGS.param_ef, FLAGS.param_efps))

    random_support = False
    total_rwd = 0.0
    act_val = 1
    try:
        obs = env.reset()
        n_step = 0
        # run this episode
        while True:
            n_step += 1
            #print('observation=', obs, 'observation_none=', obs[None])
            action = act_val#act(obs[None])[0]
            obs, rwd, done, other = env.step(action)
            print('action=', action, '; rwd=', rwd, '; step=', n_step)
            total_rwd += rwd
            if other:
                act_val = 7
            if random_support:
                act_val = random.randint(0, 8)

            if n_step == 50:
                act_val = 3
            '''
            if n_step == 20:
                act_val = 0
            elif n_step == 94:
                act_val = 1
            '''
            #print('step rwd=', rwd, ',action=', action, "obs=", obs)
            if done:
                print("game over, total_rwd=", total_rwd)
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("evaluation over")
    env.unwrapped.save_replay('evaluate')
    env.close()


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)

if __name__ == "__main__":
    app.run(main)
