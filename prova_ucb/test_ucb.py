import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from prova_ucb.enviroment import *
from prova_ucb.UCBLearner import *
from prova_ucb.gts_learner import *

T = 300
arms = np.array(([0, 1, 2, 3, 4, 5]))
n_costumers = 1000
n_arms = arms.shape[0]
n_experiments = 1000

reward_per_experiment_ucb = []
reward_per_experiment_gts = []
env = EnviromentUCB(arms, n_costumers)
for e in trange(n_experiments):
    rewards_ucb = []
    reward_gts = []
    ucb_learner = UCBLearner(n_arms)
    gts_learner = GTS_Learner(n_arms)
    for t in range(T):
        pulled_arm = ucb_learner.pull_arm()
        reward = env.round(pulled_arm)
        ucb_learner.update(pulled_arm, reward)
        rewards_ucb.append(reward)

        pulled_arm = gts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gts_learner.update(pulled_arm, reward)
        reward_gts.append(reward)

    reward_per_experiment_ucb.append(ucb_learner.collected_rewards)
    reward_per_experiment_gts.append(gts_learner.collected_rewards)

optimal_reward = env.opt_reward()
# print(reward_per_experiment)
# print(optimal_reward)
plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment_ucb, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment_gts, axis=0)), 'b')
plt.show()
