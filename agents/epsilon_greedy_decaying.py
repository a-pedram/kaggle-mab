
#decay_rates = [0.99, 0.98, 0.97, 0.96, 0.95]

last_bandit = -1
total_reward = 0

sums_of_reward = None
numbers_of_selections = None
    
def agent(observation, configuration):    
    global sums_of_reward, numbers_of_selections, last_bandit, total_reward

    if observation.step == 0:
        numbers_of_selections = [0] * configuration.banditCount
        sums_of_reward = [0] * configuration.banditCount

    if last_bandit > -1:
        reward = observation.reward - total_reward
        sums_of_reward[last_bandit] += reward
        total_reward += reward

    eps_2 = 0

    bandit = 0
    max_upper_bound = 0
    for i in range(0, configuration.banditCount):
        if (numbers_of_selections[i] > 0):
            eps_2 += 1
            decay = 0.99*(0.001*eps_2) ** numbers_of_selections[i]
            upper_bound = decay * sums_of_reward[i] / numbers_of_selections[i]
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound and last_bandit != i:
            max_upper_bound = upper_bound
            bandit = i
            last_bandit = bandit

    numbers_of_selections[bandit] += 1

    if bandit is None:
        bandit = 0

    return bandit

