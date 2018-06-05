from gym.envs.registration import register

register(
    id='SC2GYMENV-v0',
    entry_point='sc2scout.envs:SC2GymEnv',
)

