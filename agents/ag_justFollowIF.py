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

def decayed_probab(n_ones, n_zeros, his_n):
    global sp, spp
    ps = sp**n_ones * spp **n_zeros
    limit = int(1000 * decay_rate**(his_n + n_ones + n_zeros + 1 ) )+1
    cdfBeta = np.cumsum(ps[:limit] )
    place = np.argmin(np.abs(cdfBeta - cdfBeta[-1] / 2 ))
    return sp[place] + np.random.rand() * 0.0001

myProbabs = None
his_bandit_rec = None
his_rec_index = None
kExp = 0.1
n_lookback = 60
max_hit = 66
probab_ratio = 1.5
n_decision = 7
def new_bandit(step):
    global myProbabs, his_low_chances
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    # n_last_move = (his_last_moves == his_last_bandit).sum()
    
    myProbabs[last_bandit] = decayed_probab(wins[last_bandit],losses[last_bandit],his_hits[last_bandit])
    myProbabs[his_last_bandit] = decayed_probab(wins[his_last_bandit],losses[his_last_bandit],his_hits[his_last_bandit])
    my_max = myProbabs.max()
    my_choice = np.argmax(myProbabs)
    if my_max / myProbabs[his_last_bandit] > probab_ratio  and my_hits[his_last_bandit] > n_decision and his_hits[his_last_bandit]>1:
        return np.argmax(myProbabs)
    else:
        #if his_hits[his_last_bandit] >=2:
        return his_last_bandit
        #return np.argmax(myProbabs)

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs, his_scores
    global last_bandit, total_reward,  record_index, wins, losses, his_rec_index, his_bandit_rec
    if obs.step == 0:        
        total_reward = 0 
        his_record = np.zeros(n_rounds, 'int')
        his_hits = np.zeros(conf.banditCount, 'int')
        his_scores = np.zeros(bandit_count) + 0.5
        his_bandit_rec = np.ones([bandit_count, 2000], 'int') * -1
        his_rec_index = np.zeros(bandit_count, 'int')
        
        myProbabs = np.random.rand(bandit_count)* 0.001 + 0.5
        my_record = np.zeros(n_rounds, 'int')
        my_hits = np.zeros(conf.banditCount, 'int')
        bandits_record = np.zeros([conf.banditCount, 600], 'int')
        record_index = np.zeros(conf.banditCount,'int')
        wins = np.zeros(conf.banditCount, 'int')
        losses = np.zeros(conf.banditCount, 'int')

        bandit = np.random.randint(conf.banditCount)
        last_bandit = bandit
        return bandit
    if obs.lastActions[0] == last_bandit:
        his_action = obs.lastActions[1]
    else:
        his_action = obs.lastActions[0]
    his_record[obs.step-1] = his_action
    his_hits[his_action] += 1
    his_bandit_rec[his_action,his_rec_index[his_action]] = obs.step
    his_rec_index[his_action] +=1
    my_hits[last_bandit] += 1
    my_record[obs.step-1] = last_bandit
    if obs.reward > total_reward:
        total_reward += 1
        bandits_record[last_bandit, record_index[last_bandit]] = 1
        wins[last_bandit] += 1
    else:
        bandits_record[last_bandit, record_index[last_bandit]] = 0
        losses[last_bandit] +=1
    record_index[last_bandit] += 1
    bandit = int(new_bandit(obs.step))
    last_bandit = bandit 
    return bandit
