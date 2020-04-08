import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from pricing.environments import EnvironmentUCB
from pricing.learners.UCBLearner import UCBLearner
from pricing.learners.ts_learner import TSLearner

prices = [63, 76, 10, 8, 53, 21]
T = 100
arms = [0, 1, 2, 3, 4, 5]
n_arms = len(arms)
n_experiments = 50



reward_per_experiment_ucb = []
reward_per_experiment_ts = []
env = EnvironmentUCB(arms=arms, prices=prices)

regret_per_experiment_ucb = []
regret_per_experiment_ts = []

for e in trange(n_experiments):
    ucb_learner = UCBLearner(n_arms, prices)
    ts_learner = TSLearner(n_arms, prices)

    cumulative_regret_ucb = 0
    cumulative_regret_ts = 0

    regrets_ucb_per_timestep = []
    regrets_ts_per_timestep = []

    for t in range(T):
        clicks = round(np.random.normal(10, 0.1))
        # clicks = 10

        for _ in range(clicks):
            #UCB
            pulled_arm = ucb_learner.pull_arm()
            reward = env.round(pulled_arm)
            ucb_learner.update(pulled_arm, reward)

            instantaneous_regret = env.get_inst_regret(pulled_arm)
            cumulative_regret_ucb += instantaneous_regret

            #TS
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            instantaneous_regret = env.get_inst_regret(pulled_arm)
            cumulative_regret_ts += instantaneous_regret

        regrets_ucb_per_timestep.append(cumulative_regret_ucb)
        regrets_ts_per_timestep.append(cumulative_regret_ts)


    regret_per_experiment_ts.append(regrets_ts_per_timestep)
    regret_per_experiment_ucb.append(regrets_ucb_per_timestep)

        # ucb_regret_per_timestep.append(np.mean(regret_ucb))
        # gts_regret_per_timestep.append(np.mean(regret_gts))


    # regret_per_experiment_gts.append(gts_regret_per_timestep)
    # regret_per_experiment_ucb.append(ucb_regret_per_timestep)


# print(reward_per_experiment)
# print(optimal_reward)
plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
# TODO check if better plotting expected regret or mean regret
plt.plot(np.mean(regret_per_experiment_ucb, axis=0), 'r')
plt.plot(np.mean(regret_per_experiment_ts, axis=0), 'b')
plt.legend(['UCB1', "TS"])
plt.show()
