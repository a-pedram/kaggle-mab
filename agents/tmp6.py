import numpy as np

decay_rate = 0.97
n_rounds = 2000
bandit_count = 100

total_reward = None 
last_bandit = None
his_hits = None
his_record = None
my_record = None
my_hits = None
wins = None
losses = None
bandits_record = None
record_index = None
x1 = None
x2 = None

sp = np.linspace(0,1,1000)
spp = 1 - sp
def probab(n_ones,n_zeros):
    global sp, spp
    cdfBeta = np.cumsum(sp**n_ones * spp **n_zeros )
    place = np.argmin(np.abs(cdfBeta - cdfBeta[999] / 2 ))
    return sp[place] + np.random.rand() * 0.0001

myProbabs = np.ones(bandit_count) * 0.5
def new_bandit(step):
    global bandits_record, his_hits, his_record, last_bandit, myProbabs,my_hits 
    not_chosens = (my_hits==0)
    if not_chosens.any():
        scores_from_him = np.zeros(bandit_count)
        scores_from_him += (his_hits - 1)**2 - 1
        if his_hits[his_record[step]] == 1:
            scores_from_him[his_record[step]] += 1
        scores_from_him[my_hits!=0] = 0
        his_winner_not_chosen = int( np.argmax(scores_from_him))
    
    myProbabs[last_bandit] = probab(wins[last_bandit],losses[last_bandit])
    scores = myProbabs + 9 / (losses**1.5 +1)
    max_score = np.max(scores)
    bests = np.where(np.abs(scores - max_score) < 0.01)[0]
    winner = int(bests[np.random.randint(bests.shape[0])])
    if my_hits[winner]==0 and myProbabs[last_bandit] > .77 :
        bests = np.where(myProbabs > .77 )[0]
        winner = int(bests[np.random.randint(bests.shape[0])])
    if my_hits[ winner] == 0 and my_hits[his_winner_not_chosen]==0:
        #print('my winner',winner,my_hits[winner] ,'his winner',his_winner_not_chosen)
        winner = his_winner_not_chosen
    '''print('my hits',my_hits)
    print('scores',scores)
    print('bests', bests)
    input()''' 
    return winner

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs
    global last_bandit, total_reward,  record_index, wins, losses
    if obs.step == 0:        
        total_reward = 0 
        his_record = np.zeros(n_rounds,'int')
        his_hits = np.zeros(conf.banditCount,'int')
        
        myProbabs = np.ones(bandit_count) * 0.5
        my_record = np.zeros(n_rounds,'int')
        my_hits = np.zeros(conf.banditCount,'int')
        bandits_record = np.zeros([200, conf.banditCount],'int')
        record_index = np.zeros(conf.banditCount,'int')
        wins = np.zeros(conf.banditCount)
        losses = np.zeros(conf.banditCount)

        bandit = np.random.randint(conf.banditCount)
        last_bandit = bandit
        return bandit
    if obs.lastActions[0] == last_bandit:
        his_action = obs.lastActions[1]
    else:
        his_action = obs.lastActions[0]
    his_record[obs.step] = his_action
    his_hits[his_action] += 1
    my_hits[last_bandit] += 1
    my_record[obs.step] = last_bandit
    if obs.reward > total_reward:
        total_reward += 1
        bandits_record[record_index[last_bandit], last_bandit] = 1
        wins[last_bandit] += 1
    else:
        bandits_record[record_index[last_bandit], last_bandit] = 0
        losses[last_bandit] +=1
    record_index[last_bandit] += 1
    bandit = new_bandit(obs.step)
    last_bandit = bandit 
    return bandit
