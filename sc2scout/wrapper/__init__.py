from sc2scout.wrapper.wrapper_factory import make, register, model
from sc2scout.wrapper.explore_enemy import ExploreMakerV0, \
ExploreMakerV1, ExploreMakerV2, ExploreMakerV3, ExploreMakerV4, \
ExploreMakerV5, ExploreMakerV6, ExploreMakerV7, ExploreMakerV8, \
ExploreMakerV9, ExploreMakerV10

register('explore_v0', ExploreMakerV0())
register('explore_v1', ExploreMakerV1())
register('explore_v2', ExploreMakerV2())
register('explore_v3', ExploreMakerV3())
register('explore_v4', ExploreMakerV4())
register('explore_v5', ExploreMakerV5())
register('explore_v6', ExploreMakerV6())
register('explore_v7', ExploreMakerV7())
register('explore_v8', ExploreMakerV8())
register('explore_v9', ExploreMakerV9())
register('explore_v10', ExploreMakerV10())

