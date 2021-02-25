import numpy as np

decay_rate = 0.97

total_reward = None 
last_bandit = None
his_overall_record = None
total_steps = 2000

record_size = 7
bandits_record = None
record_index = None
my_hits_p_1 = None
expQ = 0

def new_bandit(step):
    global bandits_record, his_overall_record, last_bandit 
    exploration_score = 1 / my_hits_p_1
    scores = bandits_record.sum(axis=0)  +  expQ * exploration_score
    winner = np.argmax(scores)    
    return int(winner)

def agent(obs, conf):
    global bandits_record, his_overall_record
    global last_bandit, total_reward, my_hits_p_1, record_index
    if obs.step == 0:        
        his_overall_record = np.zeros(conf.banditCount)
        total_reward = 0 
        bandits_record = np.zeros([record_size, conf.banditCount])
        record_index = np.zeros(conf.banditCount,'int')
        my_hits_p_1 = np.ones(conf.banditCount)
        bandit = np.random.randint(conf.banditCount)
        last_bandit = bandit
        return bandit
    if obs.lastActions[0] == last_bandit:
        his_action = obs.lastActions[1]
    else:
        his_action = obs.lastActions[0]
    his_overall_record[his_action] += 1
    my_hits_p_1[last_bandit] += 1
    if obs.reward > total_reward:
        total_reward += 1
        bandits_record[record_index[last_bandit], last_bandit] = 1        
    else:
        bandits_record[record_index[last_bandit], last_bandit] = -1
    record_index[last_bandit] = (record_index[last_bandit] + 1) % record_size 
    bandit = new_bandit(obs.step)
    last_bandit = bandit 
    return bandit
