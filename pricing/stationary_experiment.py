import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import trange
from pricing.conversion_rate import ProductConversionRate
from pricing.environments import StationaryEnvironment
from pricing.learners.UCBLearner import UCBLearner
from pricing.learners.ts_learner import TSLearner

PRICES = [2,3,4,5,6,7,8,9,10,11]
T = 100
N_ARMS = 10
N_EXPERIMENTS = 50


# Load product conversion rate curve information
with open("products/products.json", 'r') as productfile:
    p_info = json.load(productfile)
    productfile.close()

    p_2 = p_info["products"][1]
    p_id = p_2["product_id"]
    seasons = p_2["seasons"]
    s_id = seasons[0]["season_id"]
    y = seasons[0]["y_values"]
    curve = ProductConversionRate(p_id, s_id, N_ARMS, y)


# Support variables
env = StationaryEnvironment(prices=PRICES, curve=curve)

reward_per_experiment_ucb = []
reward_per_experiment_ts = []

regret_per_experiment_ucb = []
regret_per_experiment_ts = []

for _ in trange(N_EXPERIMENTS):
    # Instantiate the learners
    ucb_learner = UCBLearner(N_ARMS, PRICES)
    ts_learner = TSLearner(PRICES)

    # Reset support variables
    cumulative_regret_ucb = 0
    cumulative_regret_ts = 0

    regrets_ucb_per_timestep = []
    regrets_ts_per_timestep = []

    # Evaluate performance over the time horizon
    for t in range(T):
        clicks = round(np.random.normal(10, 0.1))

        # Choose a price for each user and compute reward
        for _ in range(clicks):
            #UCB learner
            pulled_arm = ucb_learner.pull_arm()
            reward = env.round(pulled_arm)
            ucb_learner.update(pulled_arm, reward)

            instantaneous_regret = env.get_inst_regret(pulled_arm)
            cumulative_regret_ucb += instantaneous_regret

            #TS learner
            pulled_arm = ts_learner.pull_arm()
            reward = env.round(pulled_arm)
            ts_learner.update(pulled_arm, reward)

            instantaneous_regret = env.get_inst_regret(pulled_arm)
            cumulative_regret_ts += instantaneous_regret

        regrets_ucb_per_timestep.append(cumulative_regret_ucb)
        regrets_ts_per_timestep.append(cumulative_regret_ts)

    regret_per_experiment_ts.append(regrets_ts_per_timestep)
    regret_per_experiment_ucb.append(regrets_ucb_per_timestep)

# Plot the regret over time
plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
# TODO check if better plotting expected regret or mean regret
plt.plot(np.mean(regret_per_experiment_ucb, axis=0), 'r')
plt.plot(np.mean(regret_per_experiment_ts, axis=0), 'b')
plt.legend(['UCB1', "TS"])
plt.show()
