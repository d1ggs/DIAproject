import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from pricing.environments import StationaryEnvironment
from pricing.learners.ts_learner import TSLearner
from pricing import GreedyLearner

n_arms = 4
p = np.array([0.15, 0.1, 0.1, 0.35])
opt = p[3]

T = 100

n_experiments = 300

ts_rewards_per_experiment = []
gr_rewards_per_experiment = []

for e in trange(n_experiments):
    env = StationaryEnvironment(n_arms=n_arms, probabilities=p)
    ts_learner = TSLearner()
    gr_learner = GreedyLearner(n_arms=n_arms)

    ts_rewards = []
    gr_rewards = []
    for t in range(T):
        # Loop for both learners simultaneously

        # Thompson Sampling Learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)
        ts_rewards.append(reward)

        # Greedy Learner
        pulled_arm = gr_learner.pull_arm()
        reward = env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)
        gr_rewards.append(reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt - gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()

