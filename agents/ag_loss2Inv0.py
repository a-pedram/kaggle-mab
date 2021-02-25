
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
last_ones = None
n_last_ones = 20
bandits_record = None
record_index = None
x1 = None
x2 = None

sp = np.linspace(0,1,1000)
spp = 1 - sp
n_lookback = 60

def decayed_probab(n_ones, n_zeros, his_n):
    global sp, spp
    ps = sp**n_ones * spp **n_zeros
    limit = int(1000 * decay_rate**(his_n + n_ones + n_zeros + 1 ) )+1
    cdfBeta = np.cumsum(ps[:limit] )
    place = np.argmin(np.abs(cdfBeta - cdfBeta[-1] / 2 ))
    return sp[place] + np.random.rand() * 0.0001

myProbabs = np.ones(bandit_count) * 0.5
def new_bandit(step):
    global bandits_record, his_hits, his_record, last_bandit, myProbabs,my_hits, last_ones 
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    n_last_move = (his_last_moves == his_last_bandit).sum()
    
    myProbabs[last_bandit] = decayed_probab(wins[last_bandit],losses[last_bandit],his_hits[last_bandit])
    myProbabs[his_last_bandit] = decayed_probab(wins[his_last_bandit],losses[his_last_bandit],his_hits[his_last_bandit])


    scores = last_ones.sum(axis =0)
    scores = ((last_ones==1).sum(axis=0) + 1 ) /((last_ones==-1).sum(axis=0) + 1)**2
    scores[my_hits+his_hits > 66] = -99
    scores[myProbabs < .12] = -50
    winner = int(np.argmax(scores))
    return winner

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs
    global last_bandit, total_reward,  record_index, wins, losses, last_ones, n_last_ones
    if obs.step == 0:        
        total_reward = 0 
        his_record = np.zeros(n_rounds,'int')
        his_hits = np.zeros(conf.banditCount,'int')
        
        myProbabs = np.ones(bandit_count) * 0.5
        my_record = np.zeros(n_rounds,'int')
        my_hits = np.zeros(conf.banditCount,'int')
        bandits_record = np.zeros([600, conf.banditCount],'int')
        last_ones = np.zeros([n_last_ones, conf.banditCount],'int')
        record_index = np.zeros(conf.banditCount, 'int')
        wins = np.zeros(conf.banditCount, 'int')
        losses = np.zeros(conf.banditCount, 'int')

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
        last_ones[record_index[last_bandit]% n_last_ones ,last_bandit] = 1
    else:
        bandits_record[record_index[last_bandit], last_bandit] = 0
        losses[last_bandit] +=1
        last_ones[record_index[last_bandit]% n_last_ones ,last_bandit] = -1
    record_index[last_bandit] += 1
    bandit = int(new_bandit(obs.step))
    last_bandit = bandit 
    return bandit
