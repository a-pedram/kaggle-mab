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

def decayed_probab(n_ones, n_zeros, his_n):
    global sp, spp
    ps = sp**n_ones * spp **n_zeros
    limit = int(1000 * decay_rate**(his_n + n_ones + n_zeros + 1 ) )+1
    cdfBeta = np.cumsum(ps[:limit] )
    place = np.argmin(np.abs(cdfBeta - cdfBeta[-1] / 2 ))
    return sp[place] + np.random.rand() * 0.0001

myProbabs = None
his_scores = None
his_bandit_rec = None
his_rec_index = None
kExp = 0.222
kHisSco = 0.1
n_lookback = 60
def new_bandit(step):
    global myProbabs
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    n_last_move = (his_last_moves == his_last_bandit).sum()
    
    myProbabs[last_bandit] = decayed_probab(wins[last_bandit],losses[last_bandit],his_hits[last_bandit])
    myProbabs[his_last_bandit] = decayed_probab(wins[his_last_bandit],losses[his_last_bandit],his_hits[his_last_bandit])
    
    if n_last_move > 1 and my_hits[his_last_bandit] <= 2:
        return his_last_bandit
    
    if n_last_move == 1 and his_hits[his_last_bandit] > 1 and myProbabs[his_last_bandit] > 0.45 and step <1500 :
        #the last 2 conditions are to avoid problems as such posed by uniform agent
        myProbabs[his_last_bandit] -= .25

    his_scores = np.zeros(bandit_count)
    if step > n_lookback:
        his_b, counts = np.unique(his_last_moves, return_counts=True)
        his_scores[his_b] = counts
        for i in range(his_last_moves.shape[0]):
            his_scores[his_last_moves[i]] += i
        his_scores = his_scores/ np.max(his_scores)

    exploratio_score = np.zeros(bandit_count)
    exploratio_score[his_hits == 0] = -0.25
    exploratio_score[his_hits == 1] = -0.5
    exploratio_score += 1 /(my_hits + 1)

    scores = myProbabs + kExp * exploratio_score + kHisSco * his_scores * his_scores + .1/(losses+1)

    scores[(my_hits+his_hits) > 84] = -20
    winner = np.argmax(scores)
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
