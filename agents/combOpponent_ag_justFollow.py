import numpy as np
import pandas as pd
import random, os, datetime, math

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
last_rewardMine = None
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
max_hit = 62
unlessMyProbab = 0.15
def new_bandit(step):
    global myProbabs
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    #n_last_move = (his_last_moves == his_last_bandit).sum()
    
    myProbabs[last_bandit] = decayed_probab(wins[last_bandit],losses[last_bandit],his_hits[last_bandit])
    myProbabs[his_last_bandit] = decayed_probab(wins[his_last_bandit],losses[his_last_bandit],his_hits[his_last_bandit])

    scores = myProbabs.copy()
    hits = my_hits + his_hits
    if myProbabs[his_last_bandit] > unlessMyProbab and hits[his_last_bandit] <max_hit :
        return his_last_bandit
    if his_hits[his_last_bandit] == 1:
        if last_rewardMine == 1:
            return last_bandit

    scores[hits > max_hit] = -1000
    winner = np.argmax(scores)
    return winner





total_reward = 0
bandit_dict = {}

def set_seed(my_seed=42):
    os.environ['PYTHONHASHSEED'] = str(my_seed)
    random.seed(my_seed)
    np.random.seed(my_seed)

def get_next_bandit():
    best_bandit = 0
    best_bandit_expected = 0
    for bnd in bandit_dict:
        expect = (bandit_dict[bnd]['win'] - bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'] - (bandit_dict[bnd]['opp']>0)*1.5) \
                 / (bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp']) \
                * math.pow(0.97, bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'])
        if expect > best_bandit_expected:
            best_bandit_expected = expect
            best_bandit = bnd
    return best_bandit

my_action_list = []
op_action_list = []


# def multi_armed_probabilities(observation, configuration):
def agentOpp(observation, configuration):
    global total_reward, bandit_dict

    my_pull = random.randrange(configuration.banditCount)
    if 0 == observation.step:
        set_seed()
        total_reward = 0
        bandit_dict = {}
        for i in range(configuration.banditCount):
            bandit_dict[i] = {'win': 1, 'loss': 0, 'opp': 0}
    else:
        last_reward = observation.reward - total_reward
        total_reward = observation.reward
        
        my_idx = observation.agentIndex
        my_last_action = observation.lastActions[my_idx]
        op_last_action = observation.lastActions[1-my_idx]
        
        my_action_list.append(my_last_action)
        op_action_list.append(op_last_action)
        
        if 0 < last_reward:
            bandit_dict[my_last_action]['win'] = bandit_dict[my_last_action]['win'] +1
        else:
            bandit_dict[my_last_action]['loss'] = bandit_dict[my_last_action]['loss'] +1
        bandit_dict[op_last_action]['opp'] = bandit_dict[op_last_action]['opp'] +1
        
        if last_reward > 0:
            my_pull = my_last_action
        else:
            if observation.step >= 4:
                if (my_action_list[-1] == my_action_list[-2]) and (my_action_list[-1] == my_action_list[-3]):
                    if random.random() < 0.5:
                        my_pull = my_action_list[-1]
                    else:
                        my_pull = get_next_bandit()
                else:
                    my_pull = get_next_bandit()
            else:
                my_pull = get_next_bandit()
    
    return my_pull

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs, his_scores, last_rewardMine
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
        return agentOpp(obs, conf)
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
        last_rewardMine = 1
    else:
        bandits_record[last_bandit, record_index[last_bandit]] = 0
        losses[last_bandit] +=1
        last_rewardMine =0 
    record_index[last_bandit] += 1
    if obs.step < 430:
        bandit = agentOpp(obs, conf)
    else:
        bandit = int(new_bandit(obs.step))
    last_bandit = bandit 
    return bandit
