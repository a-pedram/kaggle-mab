import numpy as np
from collections import Counter

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
x1 = 1 
x2 = 1

def new_bandit(step):
    global bandits_record, his_hits, his_record, last_bandit 
    length = 8
    if step > length:
        his_start_idx = step - length
    else:
        his_start_idx =0
    counts = Counter(his_record[his_start_idx:step+1])
    his_choice = counts.most_common(1)[0][0]
    count_his_choice = counts.most_common(1)[0][1]

    scores = (wins +x1 )/ (wins+losses+ x2)
    my_winner = int(np.argmax(scores))
    my_probab = scores[my_winner]
   
    if my_probab < 0.51:
        if my_hits[his_choice] < 4 and his_hits[his_choice] > 3:
            return int(his_choice)
    return int(my_winner)

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record
    global last_bandit, total_reward,  record_index, wins, losses
    if obs.step == 0:        
        total_reward = 0 
        his_record = np.zeros(n_rounds,'int')
        his_hits = np.zeros(conf.banditCount,'int')
        
        my_record = np.zeros(n_rounds,'int')
        my_hits = np.zeros(conf.banditCount,'int')
        bandits_record = np.zeros([500, conf.banditCount])
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
    bandit = new_bandit(obs.step)
    record_index[last_bandit] += 1
    last_bandit = bandit 
    return bandit
