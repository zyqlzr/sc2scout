from baselines import deepq

from pysc2 import maps
from pysc2.env import sc2_env
from sc2scout.envs import ZergEvadeEnv
from sc2scout.agents import RandomAgent
from sc2scout.wrapper import make, model
from sc2scout.wrapper.util.sc2_params import races, difficulties

from absl import app
from absl import flags
import time

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")

flags.DEFINE_integer("screen_resolution", 64,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")
flags.DEFINE_float("screen_ratio", "1.33",
                   "Screen ratio of width / height")
flags.DEFINE_string("agent_interface_format", "feature",
                    "Agent Interface Format: [feature|rgb]")

flags.DEFINE_integer("max_agent_episodes", 1, "Total agent episodes.")
flags.DEFINE_integer("max_step", 4000, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
flags.DEFINE_integer("random_seed", None, "Random_seed used in game_core.")
flags.DEFINE_string("model_dir", './model_save/evade_model/model1', "model directory")
flags.DEFINE_string("wrapper", 'evade_v0', "the name of wrapper")
flags.DEFINE_enum("agent_race", 'Z', races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", 'Z', races.keys(), "Bot's race.")
flags.DEFINE_string("difficulty", '9', "Bot's strength.")

flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", 'scout_evade', "Name of a map to use.")
flags.mark_flag_as_required("map")
flags.mark_flag_as_required("wrapper")


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    is_solved = False
    return is_solved


def main(unused_argv):
    #env = gym.make("SC2GYMENV-v0")
    #env.settings['map_name'] = 'ScoutSimple64'
    if FLAGS.model_dir is None:
        print("please input --model_dir xxxxx")
        return

    rs = FLAGS.random_seed
    if FLAGS.random_seed is None:
        rs = int((time.time() % 1) * 1000000)

    players = []
    players.append(sc2_env.Bot(races[FLAGS.bot_race], difficulties[FLAGS.difficulty]))
    players.append(sc2_env.Agent(races[FLAGS.agent_race]))

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

    env = ZergEvadeEnv(
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

    network = model(FLAGS.wrapper)#deepq.models.mlp([64, 32])
    model_dir= FLAGS.model_dir
    act = deepq.load_model(env, network, model_dir)

    try:
        obs = env.reset()
        n_step = 0
        # run this episode
        while True:
            n_step += 1
            #print('observation=', obs, 'observation_none=', obs[None])
            action = act(obs[None])[0]
            obs, rwd, done, _ = env.step(action)
            print('action=', action, '; rwd=', rwd)
            #print('step rwd=', rwd, ',action=', action, "obs=", obs)
            if done:
                print("game over")
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

