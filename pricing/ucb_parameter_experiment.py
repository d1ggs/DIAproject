import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from pricing.environments import EnvironmentUCB
from pricing.learners.UCBLearner import UCBLearner
from pricing.learners.ts_learner import TSLearner

prices = [63, 76, 10, 8, 53, 21]
T = 1000
arms = [0, 1, 2, 3, 4, 5]
n_arms = len(arms)
n_experiments = 50


c_list = [0.2, 0.3, 0.5, 0.7, 0.9, 1]

reward_per_experiment_ucb = []
reward_per_experiment_ts = []
env = EnvironmentUCB(arms=arms, prices=prices)

regret_per_experiment_ts = []

print("Evaluating Thompson Sampling")

for e in trange(n_experiments):
    ts_learner = TSLearner(n_arms, prices)

    cumulative_regret_ts = 0

    regrets_ts_per_timestep = []

    for t in range(T):
        clicks = round(np.random.normal(10, 0.1))
        # clicks = 10

        for _ in range(clicks):
            # TS
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            instantaneous_regret = env.get_inst_regret(pulled_arm)
            cumulative_regret_ts += instantaneous_regret

        regrets_ts_per_timestep.append(cumulative_regret_ts)

    regret_per_experiment_ts.append(regrets_ts_per_timestep)

regret_per_parameter_ucb = []

print("\nEvaluating UCB")

for c in tqdm(c_list):
    regret_per_experiment_ucb = []

    for e in range(n_experiments):
        ucb_learner = UCBLearner(n_arms, prices, constant=c)

        cumulative_regret_ucb = 0

        regrets_ucb_per_timestep = []

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

            regrets_ucb_per_timestep.append(cumulative_regret_ucb)


        regret_per_experiment_ucb.append(regrets_ucb_per_timestep)

    regret_per_parameter_ucb.append(np.mean(regret_per_experiment_ucb, axis=0))

labels = ["UCB, c = " + str(c) for c in c_list]
labels.append("TS")

plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
# TODO check if better plotting expected regret or mean regret
for curve in regret_per_parameter_ucb:
    plt.plot(curve)
plt.plot(np.mean(regret_per_experiment_ts, axis=0))
plt.legend(labels)
plt.show()
