from baselines import deepq

from pysc2 import maps
from pysc2.env import sc2_env
from sc2scout.envs import ZergScoutEnv
from sc2scout.wrapper import ZergScoutActWrapper, ZergScoutWrapper, \
ZergScoutRwdWrapper, SkipFrame, ZergScoutObsWrapper

from absl import app
from absl import flags
import time

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_episodes", 1, "Total agent episodes.")
flags.DEFINE_integer("max_step", 100, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("random_seed", None, "Random_seed used in game_core.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run")
flags.DEFINE_string("agent_config", "",
                    "Agent's config in py file. Pass it as python module."
                    "E.g., tstarbot.agents.dft_config")
flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("disable_fog", False, "Turn off the Fog of War.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 2, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    is_solved = False
    return is_solved


def main(unused_argv):
    #env = gym.make("SC2GYMENV-v0")
    #env.settings['map_name'] = 'ScoutSimple64'

    rs = FLAGS.random_seed
    if FLAGS.random_seed is None:
        rs = int((time.time() % 1) * 1000000)

    env = ZergScoutEnv(
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

    env = ZergScoutActWrapper(env)
    env = SkipFrame(env)
    env = ZergScoutRwdWrapper(env)
    env =ZergScoutObsWrapper(env)
    env = ZergScoutWrapper(env)

    model = deepq.models.mlp([64, 32])

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to scout_model.pkl")
    act.save("scout_model.pkl")

def entry_point():  # Needed so setup.py scripts work.
    app.run(main)

if __name__ == "__main__":
    app.run(main)
