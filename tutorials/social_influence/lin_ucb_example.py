import numpy as np
import matplotlib.pyplot as plt
from tutorials.social_influence.LinearMabEnvironment import LinearMabEnvironment
from tutorials.social_influence.LinUcbLearner import LinUcbLearner

#arm corresponds to an activated edge?
n_arms = 10
T = 1000
n_experiments = 100
lin_ucb_rewards_per_experiments = []

env = LinearMabEnvironment(n_arms=n_arms, dim=10)

for e in range(0, n_experiments):
    lin_ucb_learner = LinUcbLearner(arms_features=env.arms_features)
    #next iterate over all rounds
    for t in range(0, T):
        pulled_arms = lin_ucb_learner.pull_arm()
        reward = env.round(pulled_arms)
        lin_ucb_learner.update(pulled_arms,reward)
    lin_ucb_rewards_per_experiments.append(lin_ucb_learner.collected_rewards) #store total rewards collected

opt = env.opt()
plt.figure(0)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.cumsum(np.mean(opt - lin_ucb_rewards_per_experiments, axis = 0)), 'r') #compute the regret
plt.legend(["LinUCB"])
plt.show()