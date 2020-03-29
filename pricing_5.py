from tutorials.pricing.learners import ts_learner
from conversion_rate import DemandModel, Logistic

# TODO connect to the social influence algorithm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from tutorials.pricing.environments import StationaryEnvironment
from tutorials.pricing.learners.ts_learner import TSLearner
from tutorials.pricing.learners.greedylearner import GreedyLearner

T = 1000
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
        clicks = round(np.random.normal(100, 10))
        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm() + 1
        # print(pulled_arm)
        buyers = env.compute_buyers(clicks, pulled_arm)
        # TODO check reward
        reward = buyers * pulled_arm / (clicks * pulled_arm)
        ts_learner.update(pulled_arm - 1, reward)
        ts_rewards.append(reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)

plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["TS"])
plt.show()

