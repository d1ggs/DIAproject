import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import json

from pricing.environments import NonStationaryEnvironment
from pricing.learners.ts_learner import TSLearner
from pricing.learners.swts_learner import SWTSLearner

from pricing.conversion_rate import ProductConversionRate

from pricing.learners.UCBLearner import SWUCBLearner

N_ARMS = 10
PRICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

curves = []

with open("products/products.json", 'r') as productfile:
    p_info = json.load(productfile)
    productfile.close()

    p_2 = p_info["products"][1]
    p_id = p_2["product_id"]
    for season in p_2["seasons"]:
        s_id = season["season_id"]
        y = season["y_values"]
        curves.append(ProductConversionRate(p_id, s_id, N_ARMS, y))

#prices = [1000,1100,1200,1300,1400,1500]
arms = [0, 1, 2, 3, 4,5,6,7,8,9]


T = 100

n_experiments = 300

swucb_regrets_per_experiment = []
swts_regrets_per_experiment = []
ts_regrets_per_experiment = []

window_size = 4 * int(np.sqrt(T))

NonStationaryEnvironment(curves=curves, horizon=T, prices=PRICES).plot()

for e in trange(n_experiments):

    # Reset the environments
    ts_env = NonStationaryEnvironment(curves=curves, horizon=T, prices=PRICES)
    ts_learner = TSLearner(n_arms=N_ARMS, prices=PRICES)

    swts_env = NonStationaryEnvironment(curves=curves, horizon=T, prices=PRICES)
    swts_learner = SWTSLearner(n_arms=N_ARMS, window_size=window_size, prices=PRICES)

    swucb_env = NonStationaryEnvironment(curves=curves, horizon=T, prices=PRICES)
    swucb_learner = SWUCBLearner(n_arms=N_ARMS, horizon=T, prices=PRICES)

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