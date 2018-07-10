self._rewards = [sr.EvadeTimeReward(weight=10000),
                         sr.EvadeSpaceReward(weight=10),
                         sr.EvadeHealthReward(weight=50)]

terminal: if life<max_life/3
