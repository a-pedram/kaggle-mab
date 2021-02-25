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

max_depth = 2000
num_probabs = 1001
k_decay1 = np.zeros([max_depth, num_probabs], 'float128')
k_decay0 = np.zeros([max_depth, num_probabs], 'float128')
probabs = np.linspace(0, 1, num_probabs)
for i in range(k_decay1.shape[0]):
    k_decay1[i] = probabs * 0.97 ** i
k_decay0 = 1 - k_decay1

def estimate_probab(seq):
    raw_probab1 = k_decay1[:seq.shape[0],:][seq == 1,:]
    raw_probab0 = k_decay0[:seq.shape[0],:][seq == 0,:]
    each_probab =  raw_probab0.prod(axis = 0) * raw_probab1.prod(axis = 0)
    half_area = each_probab.sum()/ 2
    cum_sum = np.cumsum(each_probab)
    best_estimation =  np.argmin(np.abs(cum_sum - half_area))
    return (probabs[best_estimation]  *.97**seq.shape[0])

myProbabs = None
his_bandit_rec = None
his_rec_index = None
kExp = 0.1
n_lookback = 100
max_hit = 64
unlessMyProbab = 0.15
followed_for_hits = 0
followed_for_mean = 0
def new_bandit(step):
    global myProbabs, followed_for_mean, followed_for_hits
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    n_last_move = (his_last_moves == his_last_bandit).sum()
    
    myProbabs[last_bandit] = estimate_probab(bandits_record[last_bandit,:record_index[last_bandit]])
    myProbabs[his_last_bandit] = estimate_probab(bandits_record[his_last_bandit,: record_index[his_last_bandit]])

    scores = myProbabs.copy()
    hits = my_hits + his_hits
    if step == 1:
        followed_for_hits =0
        followed_for_mean = 0
    if step ==1999:
        print('followed for hist:', followed_for_hits,\
             ' followed for mean:', followed_for_mean,\
             'My Choices:', 1999 - followed_for_mean -followed_for_hits)
    if my_hits[his_last_bandit] < 6 and n_last_move > 1: # <5 60:24   <4 56:28   <6 65:20  <7 65:19
        followed_for_hits += 1
        return his_last_bandit
    else:
        # if myProbabs[his_last_bandit] > myProbabs.mean():
        if myProbabs[his_last_bandit] + 1 * myProbabs.std() > myProbabs.max():
            followed_for_mean += 1 #  .8* 59:24    1* 60:24   1.5: 57:28
            return his_last_bandit #   

    scores[hits > max_hit] = -1000
    scores = scores + 1/(losses + 1) # .1/47:37 .25/ 60:25  .5/: 61:23  1/ 60:24 
    winner = np.argmax(scores)       #  1.5 58 27
    return winner

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
        bandits_record = np.zeros([conf.banditCount, 2000], 'int')
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
    bandits_record[his_action, record_index[his_action]] = -1
    record_index[his_action] += 1
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
