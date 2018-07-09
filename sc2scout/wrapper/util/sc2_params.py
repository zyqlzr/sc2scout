from pysc2.env import sc2_env

races = {
  "R": sc2_env.Race.random,
  "P": sc2_env.Race.protoss,
  "T": sc2_env.Race.terran,
  "Z": sc2_env.Race.zerg,
}

difficulties = {
  "1": sc2_env.Difficulty.very_easy,
  "2": sc2_env.Difficulty.easy,
  "3": sc2_env.Difficulty.medium,
  "4": sc2_env.Difficulty.medium_hard,
  "5": sc2_env.Difficulty.hard,
  "6": sc2_env.Difficulty.hard,
  "7": sc2_env.Difficulty.very_hard,
  "8": sc2_env.Difficulty.cheat_vision,
  "9": sc2_env.Difficulty.cheat_money,
  "A": sc2_env.Difficulty.cheat_insane,
}

