from sc2scout.wrapper.wrapper_factory import make, register
from sc2scout.wrapper.explore_enemy import ExploreMakerV0

register('explore_v0', ExploreMakerV0())

