import numpy as np
from collections import Counter

decay_rate = 0.97
n_rounds = 2000
bandit_count = 100

total_reward = None 
last_bandit = None
last_reward = None
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

myProbabs = np.random.rand(bandit_count)* 0.001 + 0.5
n_lookback = 4
def new_bandit(step):
    global bandits_record, his_hits, his_record, last_bandit, myProbabs 
    his_choice = his_record[step-1]
    myProbabs[last_bandit] = decayed_probab(wins[last_bandit],losses[last_bandit],his_hits[last_bandit])
    myProbabs[his_choice] = decayed_probab(wins[his_choice],losses[his_choice],his_hits[his_choice])
    scores = myProbabs + 3 / (losses +1)
    if step < n_lookback:
        n_back = step
    else:
        n_back = n_lookback
    his_last_moves = his_record[step - n_back:step]
    move_counts = Counter(his_last_moves)
    his_winner_wins =  move_counts.most_common()[0][1]
    his_winner = int(move_counts.most_common()[0][0])
    # print('step',step)
    # print('his record',his_record,'his_last_moves',his_last_moves)
    # print('his choice:',his_winner,'his choice win:',his_winner_wins,'his move counts',move_counts,'his hits',his_hits)
    # print('my hits',my_hits)
    if step < 1000:
        if my_hits[his_winner] <= 3 and his_winner_wins > 1 and my_hits[his_winner] < his_hits[his_winner]:
            if his_winner == his_record[step-2]:
                # print("his winner11!!!!")
                return his_winner
            else:
                if his_winner == last_bandit and last_reward == 1 :
                    # print("his winner22!!!!")
                    return his_winner
        else:
            if his_winner == last_bandit and last_reward == 1 :
                # print("his winner333!!!!")
                return his_winner
    winner = int(np.argmax(scores))
    #winner = int(scores.argsort()[-2:][np.random.randint(2)])
    # print("My winner!!!!")
    return winner

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs
    global last_bandit, total_reward,  record_index, wins, losses, last_reward
    if obs.step == 0:        
        total_reward = 0 
        his_record = np.zeros(n_rounds, 'int')
        his_hits = np.zeros(conf.banditCount, 'int')
        
        myProbabs = np.random.rand(bandit_count)* 0.001 + 0.5
        my_record = np.zeros(n_rounds, 'int')
        my_hits = np.zeros(conf.banditCount, 'int')
        bandits_record = np.zeros([400, conf.banditCount], 'int')
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
    my_hits[last_bandit] += 1
    my_record[obs.step-1] = last_bandit
    if obs.reward > total_reward:
        total_reward += 1
        bandits_record[record_index[last_bandit], last_bandit] = 1
        wins[last_bandit] += 1
        last_reward = 1
    else:
        bandits_record[record_index[last_bandit], last_bandit] = 0
        losses[last_bandit] +=1
        last_reward = 0
    record_index[last_bandit] += 1
    bandit = new_bandit(obs.step)
    last_bandit = bandit 
    return bandit