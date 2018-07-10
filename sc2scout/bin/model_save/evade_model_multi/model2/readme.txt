self._rewards = [sr.EvadeTimeReward(weight=10000),
                         sr.EvadeSpaceReward(weight=1),
                         sr.EvadeHealthReward(weight=1)]

terminal: if life<max_life/5
sr.EvadeSpaceReward: self.w = curr_step * self.w
sr.EvadeHealthReward: self.w = curr_step * self.w
