import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from tutorials.pricing.environments import NonStationaryEnvironment
from tutorials.pricing.learners.ts_learner import TSLearner
from tutorials.pricing.learners.swts_learner import SWTSLearner

from tutorials.pricing.conversion_rate import Product1Season1, Product1Season2, Product1Season3

from tutorials.pricing.learners.UCBLearner import SWUCBLearner


n_arms = 6
curves = [Product1Season1(), Product1Season2(), Product1Season3()]
prices = [63, 76, 10, 8, 53, 21]
arms = [0, 1, 2, 3, 4, 5]

T = 100

n_experiments = 1000

swucb_regrets_per_experiment = []
swts_regrets_per_experiment = []
ts_regrets_per_experiment = []

window_size = 4 * int(np.sqrt(T))

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

        instantaneous_regret = opt_reward - prices[pulled_arm]
        cumulative_regret_ts += instantaneous_regret
        regrets_ts_per_timestep.append(cumulative_regret_ts)

        # Sliding Window Thompson Sampling

        pulled_arm = swts_learner.pull_arm()
        reward = swts_env.round(pulled_arm)
        swts_learner.update(pulled_arm, reward)

        instantaneous_regret = opt_reward - prices[pulled_arm]
        cumulative_regret_swts += instantaneous_regret
        regrets_swts_per_timestep.append(cumulative_regret_swts)

        # Sliding Window UCB

        pulled_arm = swucb_learner.pull_arm()
        reward = swucb_env.round(pulled_arm)
        swucb_learner.update(pulled_arm, reward)

        instantaneous_regret = opt_reward - prices[pulled_arm]
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
# plt.plot(np.mean(swucb_regrets_per_experiment, axis=0), 'g')
plt.legend(["TS", "SW-TS", "SW-UCB"])
plt.show()