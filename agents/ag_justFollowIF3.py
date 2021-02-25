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
myProbabs = None

def new_probab(bandit_rec_c):
    bandit_rec  = bandit_rec_c.copy()
    n = bandit_rec.shape[0]
    decay_k =np.array([.97**i for i in range(n)])
    k2 =np.array([.97**i for i in range(n, 0, -1)])
    denomin = decay_k.copy()
    denomin[bandit_rec == -1] = 0
    bandit_rec[bandit_rec== -1 ] = 0
    # print('decau_k:',decay_k)
    # print('denomin:',denomin)
    return (np.sum(bandit_rec * denomin*k2)+ 1 *.97**n) / \
             (denomin.sum() + 2 )  

his_bandit_rec = None
his_rec_index = None
kExp = 0.1
n_lookback = 60
max_hit = 66
probab_ratio = 1.5
n_decision = 7
probab_step = [.47, .40, .35, .31, .26, .24, .21, .175, .15, .13]
def new_bandit(step):
    global myProbabs, his_low_chances
    start = step - n_lookback if step >= n_lookback else 0
    his_last_moves = his_record[start:step]
    his_last_bandit = his_last_moves[-1]
    # n_last_move = (his_last_moves == his_last_bandit).sum()
    step2 = int(step / 200)
    myProbabs[last_bandit] = new_probab(bandits_record[last_bandit, 0:record_index[last_bandit]])
    myProbabs[his_last_bandit] = new_probab(bandits_record[his_last_bandit, 0:record_index[his_last_bandit]])
    if myProbabs[his_last_bandit] < .9 * probab_step[step2]  and \
        my_hits[his_last_bandit] > n_decision and \
        myProbabs.max() > (probab_step[step2] * 1.05): 
        return np.argmax(myProbabs)
    else:
        #if his_hits[his_last_bandit] >=2:
        return his_last_bandit
        #return np.argmax(myProbabs))

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
    if obs.step == 1999 : print(np.round(myProbabs * 100))
    return bandit
