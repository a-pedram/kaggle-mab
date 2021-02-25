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
x1 = 6.5
x2 = 6.75

def new_bandit(step):
    global bandits_record, his_hits, his_record, last_bandit
    '''length = 100
    if step > length:
        his_start_idx = step - 100 :
    else:
        his_start_idx =0
    his_record[his_start_idx:step+1]'''
    l1 = his_record[step]
    l2 = his_record[step-1]

    scores = (wins +x1 )/ (wins+losses+ x2)
    my_winner = int(np.argmax(scores))
    my_probab = scores[my_winner]
    winner = np.argmax(scores)

    if my_probab < 0.5:
        if l1 == l2:
            if my_probab - scores[l1] < 0.05:
                if his_hits[l1] + my_hits[l1] < 40:
                    winner = l1
        elif my_probab - scores[l1] < 0.1:
            if his_hits[l1] + my_hits[l1] < 40:
                winner = l1
        elif my_probab - scores[l2] < 0.1:
            if his_hits[l2] + my_hits[l2] < 40:
                winner = l2
    return int(winner)

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
