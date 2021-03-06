import numpy as np
from collections import defaultdict
from random import choices
from scipy.special import softmax
from scipy.optimize import minimize

def sigmoid(x, x0=10):
    return 1 / (1 + np.exp(x0-x))

def maximize(f, bound=[0,1], res=0.1, tol=1e-4):
    x = np.linspace(bound[0], bound[1], int((bound[1]-bound[0])/res)+1)
    a = x[f(x).argmax()]
    if res < tol:
        return a
    else:
        return maximize(f, [max(bound[0], a-res), min(bound[1], a+res)], res/10, tol)


num_activated, num_activated_byme, LL, total_rewards, my_last_action, p_estimate = (None, )* 6

def agent(observation, configuration):
    global num_activated, num_activated_byme, LL, total_rewards, my_last_action, p_estimate
    
    N = configuration.banditCount
    d = configuration.decayRate
    
    # initialization
    if num_activated is None:
        num_activated = np.zeros(N)
        num_activated_byme = np.zeros(N)
        LL = defaultdict(lambda: '1')
        total_rewards = 0
        my_last_action = -1
        p_estimate = np.zeros(N) + 0.5

    
    # update
    if observation.lastActions:
        num_activated[observation.lastActions[0]] += 1
        num_activated[observation.lastActions[1]] += 1
        num_activated_byme[my_last_action] += 1
        last_reward = observation.reward - total_rewards
        total_rewards = observation.reward
        LL[my_last_action] += f' * (p* {d ** (num_activated[my_last_action]-1)})' if last_reward else f' * (1 - p* {d ** (num_activated[my_last_action]-1)})'
    
    # decision
    if min(num_activated_byme) < 2:
        my_last_action = int(num_activated_byme.argmin())
        return my_last_action
    
    next_prob_estimate = np.zeros(N)
    for b in observation.lastActions:
        f = lambda p: eval(LL[b])
        best_p = maximize(f, bound=[0,1])
        r = sigmoid(num_activated_byme[b], 3)
        p_estimate[b] = best_p * r + 0.5 * (1-r)
    for b in range(N):
        next_prob_estimate[b] = p_estimate[b] * d ** num_activated[b]
    
    my_last_action = int(np.argmax(next_prob_estimate))
    return my_last_action