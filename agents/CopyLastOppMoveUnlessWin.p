
import random
import sys

class CopyLastOpponentMoveUnlessWin:
    def __init__(self, retry_winrate=5/6, retry_max_step=1000, verbose=True):
        self.retry_winrate  = retry_winrate
        self.retry_max_step = retry_max_step or sys.maxsize 
        self.verbose        = verbose
        
        self.last_reward = 0
        self.last_action = 0
        self.last_winrate = { "count": 0, "reward": 0 }

    def __call__(self, obs, conf):
        return self.agent(obs, conf)

    def winrate(self) -> float:
        winrate = self.last_winrate['reward'] / max(1, self.last_winrate['count'])
        return winrate
    
    def print_winrate(self):
        winrate = self.winrate()
        #print( f"winrate {self.last_action:02d} = {self.last_winrate['reward']:2d} / {max(1, self.last_winrate['count']):2d} = {winrate:.2f}" )
        
    
    # observation   {'remainingOverageTime': 60, 'agentIndex': 1, 'reward': 0, 'step': 0, 'lastActions': []}
    # configuration {'episodeSteps': 2000, 'actTimeout': 0.25, 'runTimeout': 1200, 'banditCount': 100, 'decayRate': 0.97, 'sampleResolution': 100}
    def agent(self, obs, conf) -> int:
        # print('observation', obs)
        # print('configuration', conf)
                
        # First round doesn't have lastActions 
        if obs.step > 0 and len(obs.lastActions): 
            self.last_winrate['count'] += 1
            
            action = None
            
            # Stop doing retry near the end of the game
            if obs.step <= self.retry_max_step:            
                # Copy opponent move unless win
                if obs.reward > self.last_reward:
                    self.last_winrate['reward'] += 1
                    action = self.last_action

                # If we have found a bandit with a high winrate, keep trying it unless over retry_max_step
                elif self.winrate() >= self.retry_winrate:
                    action = self.last_action
            
            # Else copy opponent action and reset stats
            if action is None:            
                self.last_winrate = { "count": 0, "reward": 0 }
                opponentIndex  = (obs.agentIndex + 1) % len(obs.lastActions)
                opponentAction = obs.lastActions[opponentIndex]
                action         = opponentAction
        else:
            # When in doubt, be random
            action = random.randrange(conf.banditCount) 

        self.last_action = action = int(action or 0) % conf.banditCount
        self.last_reward = obs.reward
        
        if self.verbose:
            self.print_winrate()

        return action

    
instance = CopyLastOpponentMoveUnlessWin()
def agent(obs, conf):
    return instance.agent(obs, conf)
