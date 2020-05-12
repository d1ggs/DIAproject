import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import json

from pricing.environments import NonStationaryEnvironment
from pricing.learners.ts_learner import TSLearner
from pricing.learners.swts_learner import SWTSLearner

from pricing.conversion_rate import ProductConversionRate

from pricing.learners.UCBLearner import SWUCBLearner

from pricing.const import TIME_HORIZON, PRICES, N_ARMS, N_EXPERIMENTS

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

arms = [0, 1, 2, 3, 4,5,6,7,8,9]

swucb_regrets_per_experiment = []
swts_regrets_per_experiment = []
ts_regrets_per_experiment = []

window_size = 4 * int(np.sqrt(TIME_HORIZON))

NonStationaryEnvironment(curves=curves, horizon=T, prices=PRICES).plot()

for e in trange(N_EXPERIMENTS):

    # Reset the environments
    ts_env = NonStationaryEnvironment(curves=curves, horizon=TIME_HORIZON, prices=PRICES)
    ts_learner = TSLearner(n_arms=N_ARMS, prices=PRICES)

    swts_env = NonStationaryEnvironment(curves=curves, horizon=TIME_HORIZON, prices=PRICES)
    swts_learner = SWTSLearner(n_arms=N_ARMS, horizon=TIME_HORIZON, prices=PRICES)

    swucb_env = NonStationaryEnvironment(curves=curves, horizon=TIME_HORIZON, prices=PRICES)
    swucb_learner = SWUCBLearner(n_arms=N_ARMS, horizon=TIME_HORIZON, prices=PRICES)

    regrets_swts_per_timestep = []
    regrets_swucb_per_timestep = []
    regrets_ts_per_timestep = []

    cumulative_regret_swts = cumulative_regret_swucb = cumulative_regret_ts = 0

    for t in range(TIME_HORIZON):
        opt_reward = ts_env.opt_reward()
        clicks = np.random.normal(10, 0.1)
        # clicks = 10

        for _ in range(clicks):
            # Thompson Sampling
            pulled_arm = ts_learner.pull_arm()
            reward = ts_env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            instantaneous_regret = ts_env.get_inst_regret(pulled_arm)
            cumulative_regret_ts += instantaneous_regret

            # Sliding Window Thompson Sampling

            pulled_arm = swts_learner.pull_arm()
            reward = swts_env.round(pulled_arm)
            swts_learner.update(pulled_arm, reward)

            instantaneous_regret = swts_env.get_inst_regret(pulled_arm)
            cumulative_regret_swts += instantaneous_regret

            # Sliding Window UCB

            pulled_arm = swucb_learner.pull_arm()
            reward = swucb_env.round(pulled_arm)
            swucb_learner.update(pulled_arm, reward)

            instantaneous_regret = swucb_env.get_inst_regret(pulled_arm)
            cumulative_regret_swucb += instantaneous_regret

        regrets_ts_per_timestep.append(cumulative_regret_ts)
        regrets_swucb_per_timestep.append(cumulative_regret_swucb)
        regrets_swts_per_timestep.append(cumulative_regret_swts)

        # Increase timestep
        swts_env.forward_time()
        ts_env.forward_time()
        swucb_env.forward_time()

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