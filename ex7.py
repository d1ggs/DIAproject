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
from social_influence.LinUCB.LinUCBEnvironment import LinUCBEnvironment
from social_influence.const import FEATURE_MAX, FEATURE_PARAM, SOCIAL_NAMES
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.budget_allocation import StatelessBudgetAllocation

from social_influence.LinUCB.LinUCBLearner import LinUCBLearner

from utils import seeds_to_binary

# Social influence constants
MAX_NODES = 300
TOTAL_BUDGET = 3
MAX_PROPAGATION_STEPS = 2
N_EXPERIMENTS = 5

SAVEDIR = "./plots/ex_7/"


if __name__ == "__main__":

    # Social networks support variables
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
        # Read dataset
        helper = Helper()
        dataset = helper.read_dataset(social_network + "_fixed")

        # Instantiate social network objects
        social = SocialNetwork(dataset, FEATURE_PARAM[product_index], FEATURE_MAX, max_nodes=MAX_NODES)

        # Instantiate the Monte-Carlo sampler using the activation probability matrix of the current social
        mc_sampler = MonteCarloSampling(social.get_matrix().copy())

        # Load product conversion rate curve information
        product = p_info["products"][product_index]

        social_networks.append(social)
        samplers.append(mc_sampler)
        products.append(product)

    # Instantiate the budget allocator
    budget_allocator = StatelessBudgetAllocation(TOTAL_BUDGET, monte_carlo_simulations, MAX_PROPAGATION_STEPS)

    # Learners and regret support variables
    ts_regrets_per_experiment = [[], [], []]
    original_envs = []
    original_ts_learners = []
    original_matrix_learners = [LinUCBLearner(social_networks[i].get_edge_features_matrix(),
                                     monte_carlo_simulations,
                                     MAX_PROPAGATION_STEPS,
                                     TOTAL_BUDGET)
                       for i in range(3)]

    # Instantiate an environment for each social, to be used during edge activation probability learning phase
    LinUCBEnvironments = [LinUCBEnvironment(social_networks[i].get_matrix()) for i in range(3)]

    # Instantiate an environment and a learner for each social, to be used during the pricing phase
    for i in range(3):
        # Load the product
        product = products[i]
        product_id = product["product_id"]

        # Load the season curve
        seasons = product["seasons"]
        phase = 0  # In the stationary scenario only the first phase is considered
        season_id = seasons[phase]["season_id"]
        y = seasons[phase]["y_values"]
        curve = ProductConversionRate(product_id, season_id, N_ARMS, y)
        original_envs.append(StationaryEnvironment(prices=PRICES, curve=curve))

        original_ts_learners.append(TSLearner(PRICES))

    opt_weights = []
    best_arms = []

    # Compute the optimal prices for each product
    for i in range(3):
        weight, best_arm = original_envs[i].opt_reward()
        opt_weights.append(weight)
        best_arms.append(best_arm)

    print("\nPrecomputing social influence for maximum budget...")

    budget, average_joint_influence, best_seeds = budget_allocator.joint_influence_maximization(social_networks[0].get_matrix(),
                                                                                                social_networks[1].get_matrix(),
                                                                                                social_networks[2].get_matrix(),
                                                                                                weights=opt_weights,
                                                                                                split_joint_influence=True)

    opt_clicks = [int(round(average_joint_influence[i])) for i in range(3)]


    # Main experiment loop

    print("\nPerforming experiments...")

    for e in range(N_EXPERIMENTS):
        print("\nExperiment {} of {}".format(e+1, N_EXPERIMENTS))

        # Reset the learners and the environments
        matrix_learners = copy.deepcopy(original_matrix_learners)
        ts_learners = copy.deepcopy(original_ts_learners)
        envs = copy.deepcopy(original_envs)

        cumulative_regret_ts = [0, 0, 0]
        regrets_ts_per_timestep = [[0], [0], [0]]

        # Passing time loop
        for _ in trange(TIME_HORIZON):
            # Extract the currently best performing arm
            price_weights = [ts_learners[i].get_last_best_price() for i in range(3)]

            # Extract the edge activation probability estimated matrices
            estimated_matrices = [matrix_learners[i].get_prob_matrix() for i in range(3)]

            # LinUCB candidate edge extraction
            pulled_arms = [matrix_learners[i].pull_arm() for i in range(3)]

            # Compute joint influence and budget allocation
            budget, _, seeds = budget_allocator.joint_influence_maximization(estimated_matrices[0],
                                                                             estimated_matrices[1],
                                                                             estimated_matrices[2],
                                                                             weights=price_weights)

            seeds = seeds_to_binary(seeds, MAX_NODES)

            for i in range(3):

                # Convert from node indexes seed to binary encoding
                fixed_size_seeds = seeds[i]

                # Compute social influence simulating the cascade
                seeds_vector = samplers[i].simulate_episode(fixed_size_seeds, MAX_PROPAGATION_STEPS)

                # The number of users for the pricing phase is the sum of activated nodes in social influence
                learner_clicks = int(np.sum(seeds_vector))

                # Check if the target edges of LinUCB would activate and update the learner
                target_pulled = LinUCBEnvironments[i].round(pulled_arms[i])
                matrix_learners[i].update_values(pulled_arms[i], int(target_pulled))

                # Bandit pricing

                reward_ts = 0
                reward_clairvoyant = 0

                # Choose a price for each user and compute reward
                for _ in range(learner_clicks):
                    pulled_arm = ts_learners[i].pull_arm()  # Select the price to offer
                    reward = envs[i].round(pulled_arm)  # Observe if the user buys
                    ts_learners[i].update(pulled_arm, reward)
                    reward_ts += reward * PRICES[pulled_arm]

                for _ in range(opt_clicks[i]):
                    reward_clairvoyant += envs[i].round(best_arms[i]) * PRICES[best_arms[i]]

                # Update regret
                cumulative_regret_ts[i] += reward_clairvoyant - reward_ts
                regrets_ts_per_timestep[i].append(cumulative_regret_ts[i])

        for i in range(3):
            ts_regrets_per_experiment[i].append(regrets_ts_per_timestep[i])

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
        # print(df["regret"])
        sns.lineplot(x="timestep", y="regret", data=df, hue="agent")
        plt.title(social_network + " - product " + str(product_index + 1) + " : mean regret over time")
        plt.savefig(SAVEDIR + "ex_7_{}_stationary_{}n_{}bdg_{}prop_{}ex.png".format(social_network, MAX_NODES, TOTAL_BUDGET, MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
        plt.show()


    # Cumulative plot

    labels = []
    results = []
    timesteps = []
    indexes = []

    total_regret = np.sum(ts_regrets_per_experiment, axis=0)

    for experiment, index in zip(total_regret, range(len(total_regret))):
        timesteps.extend(np.arange(len(experiment)))
        results.extend(experiment)
        indexes.extend([index] * len(experiment))
        #labels.extend(["TS"] * len(experiment))

    plt.figure()
    df = pd.DataFrame({"regret": results, "timestep": timesteps, "experiment_id": indexes, "agent": labels})
    sns.lineplot(x="timestep", y="regret", data=df)
    plt.title("Cumulative mean regret over time")
    plt.savefig(SAVEDIR + "ex_7_cumulative_{}n_{}bdg_{}prop_{}ex.png".format(MAX_NODES, TOTAL_BUDGET,
                                                                             MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
    plt.show()

