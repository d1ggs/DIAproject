import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from prova_ucb.environment import EnvironmentUCB
from prova_ucb.UCBLearner import UCBLearner
from prova_ucb.gtslearner import GTSLearner

T = 100
arms = np.array(([0, 1, 2, 3, 4, 5]))
n_customers = 1000
n_arms = arms.shape[0]
n_experiments = 300

reward_per_experiment_ucb = []
reward_per_experiment_gts = []
env = EnvironmentUCB(arms, variance=0)

optimal_reward = env.opt_reward()
regret_per_experiment_ucb = []
regret_per_experiment_gts = []

for e in trange(n_experiments):
    ucb_learner = UCBLearner(n_arms)
    gts_learner = GTSLearner(n_arms)
    regret_ucb = []
    regret_gts = []

    for t in range(T):
        clicks = round(np.random.normal(10, 0.1))
        clicks = 1000
        rewards_ucb = []
        rewards_gts = []

        ucb_pulled_arm = ucb_learner.pull_arm()
        gts_pulled_arm = gts_learner.pull_arm()

        reward = round(env.round(ucb_pulled_arm) * clicks)
        ucb_learner.update(ucb_pulled_arm, reward)
        rewards_ucb.append([reward] * clicks)

        regret_ucb.append(optimal_reward * clicks - reward)

        reward = round(env.round(gts_pulled_arm) * clicks)
        gts_learner.update(gts_pulled_arm, reward)
        rewards_gts.append([reward] * clicks)

        # ucb_reward_per_timestep.append(np.mean(rewards_ucb))
        # gts_reward_per_timestep.append(np.mean(rewards_gts))

        regret_gts.append(optimal_reward*clicks - reward)

    regret_per_experiment_gts.append(regret_gts)
    regret_per_experiment_ucb.append(regret_ucb)

    # reward_per_experiment_ucb.append(ucb_learner.collected_rewards)
    # reward_per_experiment_gts.append(gts_learner.collected_rewards)

optimal_reward = env.opt_reward()
# print(reward_per_experiment)
# print(optimal_reward)
plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(regret_per_experiment_ucb, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(regret_per_experiment_gts, axis=0)), 'b')
# plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment_ucb, axis=0)), 'r')
# plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment_gts, axis=0)), 'b')
plt.legend(['UCB1', "Gaussian Process TS"])
plt.show()
