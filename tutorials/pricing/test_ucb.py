import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from tutorials.pricing.environment import EnvironmentUCB
from tutorials.pricing.learners.UCBLearner import UCBLearner
from tutorials.pricing.learners.ts_learner import TSLearner

prices = [63, 76, 10, 8, 53, 21]
T = 100
arms = np.array(([0, 1, 2, 3, 4, 5]))
n_customers = 1000
n_arms = arms.shape[0]
n_experiments = 500

reward_per_experiment_ucb = []
reward_per_experiment_gts = []
env = EnvironmentUCB(arms, variance=0.1)

optimal_reward = env.opt_reward()
regret_per_experiment_ucb = []
regret_per_experiment_gts = []

for e in trange(n_experiments):
    ucb_learner = UCBLearner(n_arms, prices)
    gts_learner = TSLearner(n_arms, prices)
    regret_ucb = []
    regret_gts = []

    rewards_gts = []
    rewards_ucb = []

    for t in range(T):
        clicks = round(np.random.normal(10, 0.1))
        # clicks = 10
        rewards_ucb_per_timestep = []
        rewards_gts_per_timestep = []

        for _ in range(clicks):

            ucb_pulled_arm = ucb_learner.pull_arm()
            gts_pulled_arm = gts_learner.pull_arm()

            reward = round(env.round(ucb_pulled_arm))
            ucb_learner.update(ucb_pulled_arm, reward)
            rewards_ucb_per_timestep.append(reward * prices[ucb_pulled_arm])
            # regret_ucb.append(optimal_reward - reward)

            reward = round(env.round(gts_pulled_arm))
            gts_learner.update(gts_pulled_arm, reward)
            rewards_gts_per_timestep.append(reward * prices[gts_pulled_arm])
            # regret_gts.append(optimal_reward - reward)

        rewards_gts.append(np.mean(rewards_gts_per_timestep))
        rewards_ucb.append(np.mean(rewards_ucb_per_timestep))

        # ucb_regret_per_timestep.append(np.mean(regret_ucb))
        # gts_regret_per_timestep.append(np.mean(regret_gts))


    # regret_per_experiment_gts.append(gts_regret_per_timestep)
    # regret_per_experiment_ucb.append(ucb_regret_per_timestep)

    reward_per_experiment_ucb.append(rewards_ucb)
    reward_per_experiment_gts.append(rewards_gts)

optimal_reward = env.opt_reward()
# print(reward_per_experiment)
# print(optimal_reward)
plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
# plt.plot(np.cumsum(np.mean(regret_per_experiment_ucb, axis=0)), 'r')
# plt.plot(np.cumsum(np.mean(regret_per_experiment_gts, axis=0)), 'b')
# TODO check if better plotting expected regret or mean regret
plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment_ucb, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment_gts, axis=0)), 'b')
plt.legend(['UCB1', "TS"])
plt.show()
