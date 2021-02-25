
import numpy as np

decay_rate = 0.97
n_rounds = 2000
bandit_count = 100
last_reward = 0
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
bandit_last_step = None

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
n_lookback = 60
max_hit = 66
probab_ratio = 1.5
n_decision = 7
probab_step = [.47, .40, .35, .31, .26, .24, .21, .175, .15, .13]
avg_proabs = [ .5 * .97**(i* 0.02) for i in range(2000)]
nu_my_bests = 70
nu_my_best_best = 10
my_bests_index = 0
choicePath = None
def new_bandit(step):
    global myProbabs, my_bests_index, choicePath
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    n_his_last_move = (his_last_moves == his_last_bandit).sum()
    myProbabs[last_bandit] = estimate_probab(bandits_record[last_bandit,:record_index[last_bandit]])
    myProbabs[his_last_bandit] = estimate_probab(bandits_record[his_last_bandit,: record_index[his_last_bandit]])

    # print(his_last_bandit)
    # print('band rec',bandits_record[his_last_bandit,: record_index[his_last_bandit]])
    # print(his_last_moves)
    # print(n_his_last_move)
    # print('------- Mine:')
    # print(last_bandit)
    # print('band rec:',bandits_record[last_bandit,:record_index[last_bandit]])
    # print(myProbabs)
    not_chosen = (my_hits == 0)
    not_chosen_any = not_chosen.any()

    if step == 1 :  choicePath = {'m1':0, 'm2':0, 'm3':0, 'm4':0,'m5':0, 'm6':0,'exploit':0,'noLoss':0}
    if step == 1999 : print(choicePath)
    if  his_hits[his_last_bandit] > 1:
        if my_hits[his_last_bandit] <= 1:
            choicePath['m1'] += 1
            return his_last_bandit
    

    if  not not_chosen_any :
        if n_his_last_move > 1 and my_hits[his_last_bandit] <= 4:
            choicePath['m2'] += 1
            return his_last_bandit


    my_best_bests2 = np.where((losses * 2.5 < wins) & (my_hits >2))[0]
    if my_best_bests2.shape[0] > 4 :
        choicePath['noLoss'] += 1
        steps = np.zeros(my_best_bests2.shape[0] )
        for i in range( my_best_bests2.shape[0] ):
            steps[i] = bandit_last_step[my_best_bests2[i]]
        winner = my_best_bests2[np.argmin(steps)]
        return winner
    
    my_bests = np.argsort(myProbabs)[-nu_my_bests:]
    my_best_bests = my_bests[-nu_my_best_best:]

    exploit = ((losses[my_best_bests] * 2 < wins[my_best_bests]) & (my_hits[my_best_bests] >1) ).all() 
    if exploit:
        choicePath['exploit'] += 1
    else:
        if his_last_bandit in my_bests:
            if my_hits[his_last_bandit] > 3:
                choicePath['m3'] += 1
                return his_last_bandit

 
    #if np.random.rand() > .5 :
    if not_chosen_any:
        if ((not_chosen) & (his_hits==0)).any():
            choicePath['m4'] += 1
            return np.random.choice(np.where((not_chosen) & (his_hits==0))[0])
        else:
            choicePath['m5'] += 1
            return np.random.choice(np.where(not_chosen)[0])

    # winner = my_bests[-(my_bests_index % nu_my_best_best)-1]
    #winner = np.random.choice( my_bests[-nu_my_best_best:])
    my_best_bests = my_bests[-nu_my_best_best:]
    steps = np.zeros(nu_my_best_best)
    for i in range( nu_my_best_best):
        steps[i] = bandit_last_step[my_best_bests[i]]
    winner = my_best_bests[np.argmin(steps)]
    choicePath['m6'] += 1
    return winner

def agent(obs, conf):
    global bandits_record, my_record, my_hits, his_hits, his_record, myProbabs, his_scores,last_reward
    global last_bandit, total_reward,  record_index, wins, losses, his_rec_index, his_bandit_rec,bandit_last_step
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
        bandit_last_step = np.zeros(conf.banditCount, 'int')

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
        last_reward = 1
    else:
        bandits_record[last_bandit, record_index[last_bandit]] = 0
        losses[last_bandit] +=1
        last_reward =0
    record_index[last_bandit] += 1
    bandit = int(new_bandit(obs.step))
    last_bandit = bandit 
    bandit_last_step[bandit] = obs.step
    return bandit