import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import trange, tqdm
import copy

from pricing.conversion_rate import ProductConversionRate
from pricing.environments import StationaryEnvironment, NonStationaryEnvironment
from pricing.learners.ts_learner import TSLearner
from pricing.const import TIME_HORIZON, N_EXPERIMENTS, PRICES, N_ARMS
from social_influence.const import FEATURE_MAX, FEATURE_PARAM
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.budget_allocation import StatelessBudgetAllocation

from social_influence.LinUCB.InfluenceMaximizator import InfluenceMaximizator
from social_influence.LinUCB.LinUCBLearner import LinUCBLearner

# Social influence constants
MAX_NODES = 30
TOTAL_BUDGET = 15
MAX_PROPAGATION_STEPS = 10
N_EXPERIMENTS = 3

SOCIAL_NAMES = ["gplus", "facebook", "wikipedia"]
# PARAMETERS = np.array(
#     [[0.1, 0.3, 0.2, 0.2, 0.2],
#      [0.4, 0.1, 0.2, 0.2, 0.1],
#      [0.5, 0.1, 0.1, 0.1, 0.2]])  # parameters for each social

if __name__ == "__main__":

    # Simulate Social Network
    social_networks = []
    samplers = []
    products = []
    monte_carlo_simulations = 3
    n_steps_max = 5

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

    # TODO here instead of social networks matrices use matrix obtained from LinUCB
    budget_allocator = StatelessBudgetAllocation(TOTAL_BUDGET, monte_carlo_simulations, MAX_PROPAGATION_STEPS)

    print("\nPerforming experiments...")

    original_envs = []

    original_ts_learners = []

    for i in range(3):
        product = products[i]
        product_id = product["product_id"]
        seasons = product["seasons"]
        curves = []
        for season in seasons:
            y = season["y_values"]
            season_id = season["season_id"]
            curves.append(ProductConversionRate(product_id, season_id, N_ARMS, y))
        original_envs.append(NonStationaryEnvironment(prices=PRICES, curves=curves, horizon=TIME_HORIZON))
        original_ts_learners.append(TSLearner(PRICES))

    for _ in range(N_EXPERIMENTS):
        matrix_learners = [LinUCBLearner(social_networks[i].get_edge_features_matrix(), monte_carlo_simulations, MAX_PROPAGATION_STEPS, TOTAL_BUDGET) for i in range(3)]

        cumulative_regret_ts = [0, 0, 0]

        regrets_ts_per_timestep = [[], [], []]
        # regrets_ucb_per_timestep = [[], [], []]

        envs = copy.deepcopy(original_envs)

        # ucb_learners.append(UCBLearner(PRICES))
        ts_learners = copy.deepcopy(original_ts_learners)
        for _ in trange(TIME_HORIZON):
            weights = [ts_learners[i].get_last_best_price() for i in range(3)]
            social_1 = matrix_learners[0].get_prob_matrix()
            social_2 = matrix_learners[1].get_prob_matrix()
            social_3 = matrix_learners[2].get_prob_matrix()

            pulled_arms = [matrix_learners[i].pull_arm() for i in range(3)]

            budget, _, seeds = budget_allocator.joint_influence_maximization(social_1, social_2, social_3, weights=weights)

            for i in range(3):

                fixed_size_seeds = np.zeros(MAX_NODES)
                fixed_size_seeds[seeds[i].astype(int)] = 1.0
                seeds_vector, target_pulled = samplers[i].simulate_episode(fixed_size_seeds, MAX_PROPAGATION_STEPS, target_edge=pulled_arms[i])

                matrix_learners[i].update_values(pulled_arms[i], int(target_pulled))

                clicks = int(np.sum(seeds_vector[0] if seeds_vector.shape[0] == 1 else seeds_vector[1]))

                # Bandit pricing

                # Choose a price for each user and compute reward
                for _ in range(clicks):
                    # UCB learner
                    # pulled_arm = ucb_learners[i].pull_arm()
                    # reward = envs[i].round(pulled_arm)
                    # ucb_learners[i].update(pulled_arm, reward)
                    #
                    # instantaneous_regret = envs[i].get_inst_regret(pulled_arm)
                    # cumulative_regret_ucb[i] += instantaneous_regret

                    # TS learner
                    pulled_arm = ts_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    ts_learners[i].update(pulled_arm, reward)

                    instantaneous_regret = envs[i].get_inst_regret(pulled_arm)
                    cumulative_regret_ts[i] += instantaneous_regret

                # regrets_ucb_per_timestep[i].append(cumulative_regret_ucb[i])
                regrets_ts_per_timestep[i].append(cumulative_regret_ts[i])

        for i in range(3):
            ts_regrets_per_experiment[i].append(regrets_ts_per_timestep[i])
            # ucb_regrets_per_experiment[i].append(regrets_ucb_per_timestep[i])

    # Plot results

    agents = ["TS"]
    regrets = [ts_regrets_per_experiment]

    labels = []
    results = []
    timesteps = []
    indexes = []

    for social_network, product_index, regret in zip(SOCIAL_NAMES, range(3), ts_regrets_per_experiment):

        # Prepare the data structures for the dataframe
        for agent, data in zip(agents, [regret]):
            for experiment, index in zip(data, range(len(data))):
                labels.extend([agent] * len(experiment))
                timesteps.extend(np.arange(len(experiment)))
                results.extend(experiment)
                indexes.extend([index] * len(experiment))

        plt.figure()
        df = pd.DataFrame({"agent": labels, "regret": results, "timestep": timesteps, "experiment_id": indexes})
        print(df["regret"])
        sns.lineplot(x="timestep", y="regret", data=df, hue="agent")
        plt.title(social_network + " - product " + str(product_index + 1) + " : mean regret over time")
        plt.savefig("ex_7_" + social_network + ".png")
