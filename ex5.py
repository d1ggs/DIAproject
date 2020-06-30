import json
from multiprocessing import pool, Pool

import numpy as np
import pandas as pd
import seaborn as sns
import os
import time
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
import copy

from pricing.conversion_rate import ProductConversionRate
from pricing.environments import StationaryEnvironment
from pricing.learners.UCBLearner import UCBLearner
from pricing.learners.ts_learner import TSLearner
from pricing.const import *
from social_influence.const import FEATURE_MAX, FEATURE_PARAM
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.budget_allocation import CumulativeBudgetAllocation

# Social influence constants
MAX_NODES = 1000
TOTAL_BUDGET = 10
MAX_PROPAGATION_STEPS = 3
N_EXPERIMENTS = 50
TIME_HORIZON = 60

SOCIAL_NAMES = ["gplus", "email", "wikipedia"]

SAVEDIR="./plots/ex_5/"


if __name__ == "__main__":

    # Simulate Social Network
    social_networks = []
    samplers = []
    products = []
    monte_carlo_simulations = 3

    with open("pricing/products/products.json", 'r') as productfile:
        p_info = json.load(productfile)
        productfile.close()

    print("Loading social networks and products...\n")

    for social_network, product_index in zip(SOCIAL_NAMES, range(len(SOCIAL_NAMES))):

        helper = Helper()
        dataset = helper.read_dataset(social_network + "_fixed")

        social = SocialNetwork(dataset, FEATURE_PARAM[product_index], FEATURE_MAX, max_nodes=MAX_NODES)
        prob_matrix = social.get_matrix().copy()
        n_nodes = prob_matrix.shape[0]

        mc_sampler = MonteCarloSampling(prob_matrix)

        social_networks.append(social)
        samplers.append(mc_sampler)

        # Load product conversion rate curve information

        product = p_info["products"][product_index]

        products.append(product)

    ts_regrets_per_experiment = [[], [], []]
    ucb_regrets_per_experiment = [[], [], []]

    print("\nPrecomputing social influence for maximum budget...")

    budget_allocator = CumulativeBudgetAllocation(social_networks[0].get_matrix(), social_networks[1].get_matrix(), social_networks[2].get_matrix(), TOTAL_BUDGET, monte_carlo_simulations, MAX_PROPAGATION_STEPS)

    print("\nPerforming experiments...")

    original_envs = []

    original_ts_learners = []
    original_ucb_learners = []

    for i in range(3):
        product = products[i]
        product_id = product["product_id"]
        seasons = product["seasons"]
        season_id = seasons[0]["season_id"]
        y = seasons[0]["y_values"]
        curve = ProductConversionRate(product_id, season_id, N_ARMS, y)
        original_envs.append(StationaryEnvironment(prices=PRICES, curve=curve))

        original_ts_learners.append(TSLearner(PRICES))
        original_ucb_learners.append(UCBLearner(PRICES, constant=0.5))

    for _ in trange(N_EXPERIMENTS):

        cumulative_regret_ts = [0, 0, 0]
        cumulative_regret_ucb = [0, 0, 0]

        regrets_ts_per_timestep = [[], [], []]
        regrets_ucb_per_timestep = [[], [], []]

        envs = copy.deepcopy(original_envs)

        ts_learners = copy.deepcopy(original_ts_learners)
        ucb_learners = copy.deepcopy(original_ucb_learners)

        for _ in range(TIME_HORIZON):
            weights = [ts_learners[i].get_last_best_price() for i in range(3)]
            budget, _, seeds = budget_allocator.joint_influence_maximization(weights=weights)

            for i in range(3):

                fixed_size_seeds = np.zeros(MAX_NODES)
                fixed_size_seeds[seeds[i].astype(int)] = 1.0
                seeds_vector = samplers[i].simulate_episode(fixed_size_seeds, MAX_PROPAGATION_STEPS)

                clicks = int(np.sum(seeds_vector[0] if seeds_vector.shape[0] == 1 else seeds_vector[1]))

                # Bandit pricing

                # Choose a price for each user and compute reward
                for _ in range(clicks):
                    #UCB learner
                    pulled_arm = ucb_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    ucb_learners[i].update(pulled_arm, reward)


                    instantaneous_regret = envs[i].get_inst_regret(pulled_arm)
                    cumulative_regret_ucb[i] += instantaneous_regret

                    # TS learner
                    pulled_arm = ts_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    ts_learners[i].update(pulled_arm, reward)

                    instantaneous_regret = envs[i].get_inst_regret(pulled_arm)
                    cumulative_regret_ts[i] += instantaneous_regret

                regrets_ucb_per_timestep[i].append(cumulative_regret_ucb[i])
                regrets_ts_per_timestep[i].append(cumulative_regret_ts[i])

        for i in range(3):
            ts_regrets_per_experiment[i].append(regrets_ts_per_timestep[i])
            ucb_regrets_per_experiment[i].append(regrets_ucb_per_timestep[i])

    # Plot results

    agents = ["TS", "UCB"]
    regrets = [ts_regrets_per_experiment, ucb_regrets_per_experiment]

    labels = []
    results = []
    timesteps = []
    indexes = []

    print("Saving plots...")

    for social_network, product_index, ts_regret, ucb_regret in zip(SOCIAL_NAMES, range(3), ts_regrets_per_experiment, ucb_regrets_per_experiment):

        for agent, data in zip(agents, [ts_regret, ucb_regret]):

        # Prepare the data structures for the dataframe

            for experiment, index in zip(data, range(len(data))):
                labels.extend([agent] * len(experiment))
                timesteps.extend(np.arange(len(experiment)))
                results.extend(experiment)
                indexes.extend([index] * len(experiment))

        plt.figure()
        df = pd.DataFrame({"agent": labels, "regret": results, "timestep": timesteps, "experiment_id": indexes})
        # print(df["regret"])
        sns.lineplot(x="timestep", y="regret", data=df, hue="agent")
        plt.title(social_network + " - product " + str(product_index + 1) + " : mean regret over time")
        plt.savefig(SAVEDIR + "ex_5_{}_{}n_{}bdg_{}prop_{}ex.png".format(social_network, MAX_NODES, TOTAL_BUDGET, MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
        plt.show()