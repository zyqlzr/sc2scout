from sc2scout.wrapper.wrapper_factory import make, register, model
from sc2scout.wrapper.explore_enemy import ExploreMakerV0, \
ExploreMakerV2, ExploreMakerV6, ExploreMakerV8, ExploreMakerV9, ExploreMakerV10, \
ExploreMakerV12
from sc2scout.wrapper.evade_enemy import EvadeMakerV0, EvadeMakerV1
from sc2scout.wrapper.explore_target import TargetMakerV1,TargetMakerV2, \
TargetMakerV3, TargetMakerV4, TargetMakerV5, TargetMakerV6, TargetMakerV7, \
TargetMakerV8
from sc2scout.wrapper.fullgame_scout import FullGameMaker, FullGameMiniMaker, \
FullGameMiniMakerV1, FullGameMiniMakerV2, FullGameMiniMakerV3, \
FullGameMakerV1, FullGameMakerV2, FullGameMakerV3, FullGameMakerV4, \
FullGameMakerV5


register('explore_v0', ExploreMakerV0())
register('explore_v2', ExploreMakerV2())
register('explore_v6', ExploreMakerV6())
register('explore_v8', ExploreMakerV8())
register('explore_v9', ExploreMakerV9())
register('explore_v10', ExploreMakerV10())
register('explore_v12', ExploreMakerV12())
register('evade_v0', EvadeMakerV0())
register('evade_v1', EvadeMakerV1())
register('target_v1', TargetMakerV1())
register('target_v2', TargetMakerV2())
register('target_v3', TargetMakerV3())
register('target_v4', TargetMakerV4())
register('target_v5', TargetMakerV5())
register('target_v6', TargetMakerV6())
register('target_v7', TargetMakerV7())
register('target_v8', TargetMakerV8())
register('fullgame_v0', FullGameMaker())
register('fullgame_mini_v0', FullGameMiniMaker())
register('fullgame_mini_v1', FullGameMiniMakerV1())
register('fullgame_mini_v2', FullGameMiniMakerV2())
register('fullgame_mini_v3', FullGameMiniMakerV3())
register('fullgame_v1', FullGameMakerV1())
register('fullgame_v2', FullGameMakerV2())
register('fullgame_v3', FullGameMakerV3())
register('fullgame_v4', FullGameMakerV4())
register('fullgame_v5', FullGameMakerV5())


