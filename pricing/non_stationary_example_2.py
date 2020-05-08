import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from pricing.environments import NonStationaryEnvironment
from pricing.learners.ts_learner import TSLearner
from pricing.learners.swts_learner import SWTSLearner

from pricing.conversion_rate_product_2 import Product2Season1, Product2Season2, Product2Season3

from pricing.learners.UCBLearner import SWUCBLearner


n_arms = 10
prices = [1,2,3,4,5,6,7,8,9,10]
curves = [Product2Season1(n_arms),Product2Season2(n_arms),Product2Season3(n_arms)]

#prices = [1000,1100,1200,1300,1400,1500]
arms = [0, 1, 2, 3, 4,5,6,7,8,9]


T = 100

n_experiments = 300

swucb_regrets_per_experiment = []
swts_regrets_per_experiment = []
ts_regrets_per_experiment = []

window_size = 4 * int(np.sqrt(T))

NonStationaryEnvironment(arms=arms, curves=curves, horizon=T, prices=prices).plot()

for e in trange(n_experiments):

    # Reset the environments
    ts_env = NonStationaryEnvironment(arms=arms, curves=curves, horizon=T, prices=prices)
    ts_learner = TSLearner(n_arms=n_arms, prices=prices)

    swts_env = NonStationaryEnvironment(arms=arms, curves=curves, horizon=T, prices=prices)
    swts_learner = SWTSLearner(n_arms=n_arms, window_size=window_size, prices=prices)

    swucb_env = NonStationaryEnvironment(arms=arms, curves=curves, horizon=T, prices=prices)
    swucb_learner = SWUCBLearner(n_arms=n_arms, horizon=T, prices=prices)

    regrets_swts_per_timestep = []
    regrets_swucb_per_timestep = []
    regrets_ts_per_timestep = []

    cumulative_regret_swts = cumulative_regret_swucb = cumulative_regret_ts = 0

    for t in range(T):
        opt_reward = ts_env.opt_reward()
        # clicks = np.random.normal(10, 0.1)
        clicks = 10

        # Thompson Sampling
        pulled_arm = ts_learner.pull_arm()
        reward = ts_env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward)

        instantaneous_regret = ts_env.get_inst_regret(pulled_arm)
        cumulative_regret_ts += instantaneous_regret
        regrets_ts_per_timestep.append(cumulative_regret_ts)

        # Sliding Window Thompson Sampling

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

        instantaneous_regret = swts_env.get_inst_regret(pulled_arm)

        cumulative_regret_swts += instantaneous_regret
        regrets_swts_per_timestep.append(cumulative_regret_swts)

        # Sliding Window UCB

        pulled_arm = swucb_learner.pull_arm()
        reward = swucb_env.round(pulled_arm)
        swucb_learner.update(pulled_arm, reward)

        instantaneous_regret = swucb_env.get_inst_regret(pulled_arm)
        cumulative_regret_swucb += instantaneous_regret
        regrets_swucb_per_timestep.append(cumulative_regret_swucb)

    swucb_regrets_per_experiment.append(regrets_swucb_per_timestep)
    swts_regrets_per_experiment.append(regrets_swts_per_timestep)
    ts_regrets_per_experiment.append(regrets_ts_per_timestep)




# ts_instantaneous_regret = np.zeros(T)
# swts_instantaneous_regret = np.zeros(T)
# n_phases = len(p)
# phases_len = int(T/n_phases)
# opt_per_phases = p.max(axis=1)
# optimum_per_round = np.zeros(T)
#
#
# for i in range(0, n_phases):
#     optimum_per_round[i*phases_len : (i+1)*phases_len] = opt_per_phases[i]
#     ts_instantaneous_regret[i*phases_len: (i+1)*phases_len] = opt_per_phases[i] - np.mean(swucb_regrets_per_experiment, axis=0)[i * phases_len:(i + 1) * phases_len]
#     swts_instantaneous_regret[i*phases_len: (i+1)*phases_len] = opt_per_phases[i] - np.mean(swts_regrets_per_experiment, axis=0)[i * phases_len:(i + 1) * phases_len]


# plt.figure(0)
# plt.ylabel("Reward")
# plt.xlabel("t")
# plt.plot(np.mean(swucb_regrets_per_experiment, axis=0), 'r')
# plt.plot(np.mean(swts_regrets_per_experiment, axis=0), 'b')
# plt.plot(optimum_per_round, '--k')
# plt.legend(["TS", "SW-TS", "Optimum"])
# plt.show()


plt.figure(1)
plt.ylabel("Regret")
plt.xlabel("t")
plt.plot(np.mean(ts_regrets_per_experiment, axis=0), 'r')
plt.plot(np.mean(swts_regrets_per_experiment, axis=0), 'b')
#plt.plot(np.mean(swucb_regrets_per_experiment, axis=0), 'g')
plt.legend(["TS", "SW-TS", "SW-UCB"])
plt.show()