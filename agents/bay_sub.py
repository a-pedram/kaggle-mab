import numpy as np
from scipy.stats import beta

ps_a = None
post_b = None
bandit = None
total_reward = 0


def agent(observation, configuration):
    global reward_sums, total_reward, bandit, post_a, post_b
    
    n_bandits = configuration.banditCount

    if observation.step == 0:
        post_a = np.ones(n_bandits)
        post_b = np.ones(n_bandits)
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        post_a[bandit] += r + (1 - observation.step / 2000)
        post_b[bandit] += (1 - r)

    
    bound = post_a / (post_a + post_b).astype(float) + beta.std(post_a, post_b) * 3
    bandit = int(np.argmax(bound))
    
    return bandit