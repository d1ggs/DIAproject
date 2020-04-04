from tutorials.pricing.conversion_rate import DemandModel, Logistic

# TODO connect to the social influence algorithm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from tutorials.pricing.learners.ts_learner import TSLearner

T = 100
n_arms = 6
midpoint = 3

n_experiments = 300

ts_rewards_per_experiment = []

env = DemandModel(Logistic(midpoint), n_arms)
opt = np.array(env.optimal_choice())

for e in trange(n_experiments):
    env = DemandModel(Logistic(midpoint), n_arms)
    ts_learner = TSLearner(n_arms=n_arms)
    ts_rewards = []

    for t in range(T):
        clicks = 100
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm() + 1
        # print(pulled_arm)
        buyers = env.compute_buyers(clicks, pulled_arm)
        # TODO check reward
        for _ in range(buyers):
            ts_learner.update(pulled_arm - 1, 1)
            ts_rewards.append(pulled_arm)
        for _ in range(clicks - buyers):
            ts_learner.update(pulled_arm - 1, 0)
            ts_rewards.append(0)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)

plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])
plt.show()

