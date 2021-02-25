import numpy as np
from scipy.stats import beta

post_a, post_b, bandit = [None] * 3
total_reward = 0
c = 3

def agent(observation, configuration):
    global total_reward, bandit, post_a, post_b, c

    if observation.step == 0:
        post_a, post_b = np.ones((2, configuration.banditCount))
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward
        # Update Gaussian posterior
        post_a[bandit] += r
        post_b[bandit] += 1 - r
    
    bound = post_a / (post_a + post_b) + beta.std(post_a, post_b) * c
    bandit = int(np.argmax(bound))
    
    return bandit