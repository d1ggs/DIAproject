import numpy as np
import matplotlib.pyplot as plt

from prova_ucb.enviroment import *
from prova_ucb.UCBLerner import *

T = 100
arms = np.array(([1, 2, 3, 4, 5]))
n_costumers = 1000
n_arms = arms.shape[0]
n_experiments = 100

reward_per_experiment = []

for e in range(n_experiments):
    env = EnviromentUCB(arms, n_costumers)
    rewards = []
    ucb_learner = UCBLearner(n_arms)
    for t in range(T):
        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)
        rewards.append(reward)
    reward_per_experiment.append(ucb_learner.collected_rewards)
    optimal_reward = env.opt_reward()

    plt.figure()
    plt.xlabel("Time")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment, axis=0)), 'r')
    plt.show()
