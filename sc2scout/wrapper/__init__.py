from sc2scout.wrapper.wrapper_factory import make, register, model
from sc2scout.wrapper.explore_enemy import ExploreMakerV0, \
ExploreMakerV2, ExploreMakerV6, ExploreMakerV8, ExploreMakerV9, ExploreMakerV10, \
ExploreMakerV12

register('explore_v0', ExploreMakerV0())
register('explore_v2', ExploreMakerV2())
register('explore_v6', ExploreMakerV6())
register('explore_v8', ExploreMakerV8())
register('explore_v9', ExploreMakerV9())
register('explore_v10', ExploreMakerV10())
register('explore_v12', ExploreMakerV12())

