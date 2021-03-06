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
    limit = int(1000 * decay_rate**(his_n + n_ones + n_zeros + 1 ) )
    cdfBeta = np.cumsum(ps[:limit] )
    place = np.argmin(np.abs(cdfBeta - cdfBeta[-1] / 2 ))
    return sp[place] + np.random.rand() * 0.0001

myProbabs = None
his_scores = None
his_bandit_rec = None
his_rec_index = None

'''
def score_his_moves(his_rec):
    if his_rec.shape[0] == 0: return 0.5
    if his_rec.shape[0] == 1: return 0
    for i in range(his_rec.shape[0]):
        
    (bandit_count - (his_rec[-1] - his_rec[-2]))/ bandit_count'''

def score_his_moves(his_record, step):
    scores = np.zeros(bandit_count)
    start = 0 if step < 100 else step - 100
    len_rec = step - start + 1
    b, counts = np.unique(his_record, return_counts=True)
    max_count = np.max(counts)
    for i in range(len_rec):
        scores[his_record[start + i]] += (i+1)/len_rec + 1
    return scores


def new_bandit(step):
    global bandits_record, his_hits, his_record, last_bandit, myProbabs, his_rec_index 
    his_choice = his_record[step-1]
    #his_scores[his_choice] = score_his_moves(his_bandit_rec[his_choice,:his_rec_index[his_choice]]] )
    his_scores = score_his_moves(his_record, step)
    myProbabs[last_bandit] = decayed_probab(wins[last_bandit],losses[last_bandit],his_hits[last_bandit])
    myProbabs[his_choice] = decayed_probab(wins[his_choice],losses[his_choice],his_hits[his_choice])
    scores = myProbabs.copy() + 2 / (losses +1)
    if step > 10 and his_record[step-1] == my_record[step-2]:
            scores[my_record[step-2]] = -9
            winner = int(scores.argsort()[-3:][np.random.randint(3)])
    else:
        winner = int(np.argmax(scores))
    #print(scores)
    #print(winner)
    return winner

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs, his_scores
    global last_bandit, total_reward,  record_index, wins, losses, his_rec_index, his_bandit_rec 
    if obs.step == 0:        
        total_reward = 0 
        his_record = np.zeros(n_rounds, 'int')
        his_hits = np.zeros(conf.banditCount, 'int')
        his_scores = np.zeros(bandit_count) + 0.5
        his_bandit_rec = np.ones([bandit_count, 600], 'int') * -1
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
    bandit = new_bandit(obs.step)
    last_bandit = bandit 
    return bandit
