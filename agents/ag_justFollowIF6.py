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

max_depth = 4000
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
n_lookback = 60
max_hit = 66
probab_ratio = 1.5
n_decision = 7
probab_step = [.47, .40, .35, .31, .26, .24, .21, .175, .15, .13]
avg_proabs = [ .5 * .97**(i* 0.02) for i in range(2000)]
def new_bandit(step):
    global myProbabs, his_low_chances
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    # n_last_move = (his_last_moves == his_last_bandit).sum()
    step2 = int(step / 200)
    myProbabs[last_bandit] = estimate_probab(bandits_record[last_bandit,:record_index[last_bandit]])
    myProbabs[his_last_bandit] = estimate_probab(bandits_record[his_last_bandit,: record_index[his_last_bandit]])
    if myProbabs[his_last_bandit] < avg_proabs[step]  and \
        my_hits[his_last_bandit] > n_decision and \
        myProbabs.max() > (avg_proabs[step] * 1): # 1.05 58:25
        return np.argmax(myProbabs)
    else:
        if his_hits[his_last_bandit] >=2:
            return his_last_bandit
        else:
            return np.argmax(myProbabs)

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