import random

total_reward = 0
last_bandit = None
my_win_record = None
my_loss_record = None
my_chances = None
def agent(observation, configuration):
    global my_chances, my_loss_record, my_win_record
    global last_bandit, total_reward
    if observation.step == 0:        
        total_reward = 0
        my_win_record = [8] * configuration.banditCount
        my_loss_record = [1] * configuration.banditCount
        my_chances = [8/9] * configuration.banditCount
        bandit = random.randrange(configuration.banditCount)
        last_bandit = bandit
        return bandit
    if observation.reward > total_reward:
        total_reward += 1
        my_win_record[last_bandit] += 1
        my_chances[last_bandit] = my_win_record[last_bandit] / (my_win_record[last_bandit] + my_loss_record[last_bandit])
        bandit = last_bandit        
    else:
        my_loss_record[last_bandit] += 1
        my_chances[last_bandit] = my_win_record[last_bandit] / (my_win_record[last_bandit] + my_loss_record[last_bandit])
        bandit = my_chances.index(max(my_chances))        
        last_bandit = bandit
    #print(my_chances)
    #print(bandit)
    return bandit
