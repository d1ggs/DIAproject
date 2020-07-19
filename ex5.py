import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
import copy

from pricing.conversion_rate import ProductConversionRate
from pricing.environments import StationaryEnvironment
from pricing.learners.UCBLearner import UCBLearner
from pricing.learners.ts_learner import TSLearner
from pricing.const import PRODUCT_FILE, N_ARMS, PRICES
from social_influence.const import FEATURE_MAX, FEATURE_PARAM, SOCIAL_NAMES
from social_influence.helper import Helper
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.social_setup import SocialNetwork
from social_influence.budget_allocation import CumulativeBudgetAllocation

# Overwritten constants
from utils import seeds_to_binary

TOTAL_BUDGET = 5
MAX_PROPAGATION_STEPS = 3
N_EXPERIMENTS = 20
TIME_HORIZON = 120
MC_SIMULATIONS = 5
MAX_NODES = 500

SAVEDIR = "./plots/ex_5/"

if __name__ == "__main__":

    social_networks = []
    samplers = []
    products = []

    # Load the products information
    with open(PRODUCT_FILE, 'r') as productfile:
        p_info = json.load(productfile)
        productfile.close()

    for product_index in range(3):
        product = p_info["products"][product_index]
        products.append(product)

    print("Loading social networks and products...\n")

    # Social network and sampler allocation
    for social_network, index in zip(SOCIAL_NAMES, range(len(SOCIAL_NAMES))):
        helper = Helper()
        dataset = helper.read_dataset(social_network + "_fixed")

        # Each social network has a number of nodes equal to MAX_NODES
        social = SocialNetwork(dataset, FEATURE_PARAM[index], FEATURE_MAX, max_nodes=MAX_NODES)

        mc_sampler = MonteCarloSampling(social.get_matrix().copy())

        samplers.append(mc_sampler)
        social_networks.append(social)

    ts_regrets_per_experiment = [[], [], []]
    ucb_regrets_per_experiment = [[], [], []]

    original_envs = []
    original_ts_learners = []
    original_ucb_learners = []

    # Conversion rate curves generation, Stationary environments and Learners initial setting
    for i in range(3):
        product = products[i]
        product_id = product["product_id"]

        seasons = product["seasons"]
        phase = 0  # In the stationary scenario only the first phase is considered
        season_id = seasons[phase]["season_id"]
        y = seasons[phase]["y_values"]
        curve = ProductConversionRate(product_id, season_id, N_ARMS, y)
        original_envs.append(StationaryEnvironment(prices=PRICES, curve=curve))

        original_ts_learners.append(TSLearner(PRICES))
        original_ucb_learners.append(UCBLearner(PRICES, constant=0.5))

    opt_weights = []
    best_arms = []

    # Compute the optimal prices for each product
    for i in range(3):
        weight, best_arm = original_envs[i].opt_reward()
        opt_weights.append(weight)
        best_arms.append(best_arm)

    # Using the optimal arms, their rewards are used to run the joint influence maximisation, in order to obtain
    # the optimal number of clicks per timestep. It is computed only one time because one-phase setting
    # has only one optimal price.
    budget_allocator = CumulativeBudgetAllocation(social_networks[0].get_matrix(), social_networks[1].get_matrix(),
                                                  social_networks[2].get_matrix(), TOTAL_BUDGET,
                                                  MC_SIMULATIONS, MAX_PROPAGATION_STEPS, verbose=False)

    print("\nPrecomputing social influence for maximum budget...")
    budget, average_joint_influence, best_seeds = budget_allocator.joint_influence_maximization(weights=opt_weights,
                                                                                                split_joint_influence=True)
    opt_clicks = [int(round(average_joint_influence[i])) for i in range(3)]

    print("\nPerforming experiments...")

    for e in range(N_EXPERIMENTS):
        print("\nExperiment {} of {}".format(e+1, N_EXPERIMENTS))

        # Initialization of the regrets
        cumulative_regret_ts = [0, 0, 0]
        cumulative_regret_ucb = [0, 0, 0]

        regrets_ts_per_timestep = [[0], [0], [0]]
        regrets_ucb_per_timestep = [[0], [0], [0]]

        # Generation of new environment and learners for the current experiment
        envs = copy.deepcopy(original_envs)
        ts_learners = copy.deepcopy(original_ts_learners)
        ucb_learners = copy.deepcopy(original_ucb_learners)

        for _ in trange(TIME_HORIZON):

            # Compute the joint inflence maximisation using the current best prices as weights for each social network
            weights = [ts_learners[i].get_last_best_price() for i in range(3)]
            budget, _, seeds = budget_allocator.joint_influence_maximization(weights=weights,
                                                                                           split_joint_influence=True)

            seeds = seeds_to_binary(seeds, MAX_NODES)
            # Social networks loop
            for i in range(3):
                # Convert from node indexes seed to binary encoding
                fixed_size_seeds = seeds[i]

                # Compute social influence simulating the cascade
                seeds_vector = samplers[i].simulate_episode(fixed_size_seeds, MAX_PROPAGATION_STEPS)

                # The number of clicks for the timestep is given by the social influence value = number of activated nodes
                learner_clicks = int(np.sum(seeds_vector))

                # Bandit pricing initialization
                reward_ucb = 0
                reward_ts = 0
                reward_clairvoyant = 0

                # Choose a price for each user and compute the reward
                for _ in range(learner_clicks):
                    # UCB learner
                    pulled_arm = ucb_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    ucb_learners[i].update(pulled_arm, reward)
                    reward_ucb += reward * PRICES[pulled_arm]

                    # TS learner
                    pulled_arm = ts_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    ts_learners[i].update(pulled_arm, reward)
                    reward_ts += reward * PRICES[pulled_arm]

                # Compute the reward given the optimal arm and optimal number of clicks
                for _ in range(opt_clicks[i]):
                    reward_clairvoyant += envs[i].round(best_arms[i]) * PRICES[best_arms[i]]

                # Calculate and append the cumulative regret of the timestep
                cumulative_regret_ucb[i] += reward_clairvoyant - reward_ucb
                cumulative_regret_ts[i] += reward_clairvoyant - reward_ts

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

    for social_network, product_index, ts_regret, ucb_regret in zip(SOCIAL_NAMES, range(3), ts_regrets_per_experiment,
                                                                    ucb_regrets_per_experiment):

        for agent, data in zip(agents, [ts_regret, ucb_regret]):

            # Prepare the data structures for the dataframe
            for experiment, index in zip(data, range(len(data))):
                labels.extend([agent] * len(experiment))
                timesteps.extend(np.arange(len(experiment)))
                results.extend(experiment)
                indexes.extend([index] * len(experiment))

        plt.figure()
        df = pd.DataFrame({"agent": labels, "regret": results, "timestep": timesteps, "experiment_id": indexes})
        sns.lineplot(x="timestep", y="regret", data=df, hue="agent")
        plt.title(social_network + " - product " + str(product_index + 1) + " : mean regret over time")
        plt.savefig(SAVEDIR + "ex_5_{}_{}n_{}bdg_{}prop_{}ex.png".format(social_network, MAX_NODES, TOTAL_BUDGET,
                                                                         MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
        plt.show()

    # Cumulative plot

    labels = []
    results = []
    timesteps = []
    indexes = []

    for agent, experiments in zip(agents, regrets):

        total_regret = np.sum(experiments, axis=0)

        for experiment, index in zip(total_regret, range(len(total_regret))):
            timesteps.extend(np.arange(len(experiment)))
            results.extend(experiment)
            indexes.extend([index] * len(experiment))
            labels.extend([agent] * len(experiment))

    plt.figure()
    df = pd.DataFrame({"regret": results, "timestep": timesteps, "experiment_id": indexes, "agent": labels})
    sns.lineplot(x="timestep", y="regret", data=df, hue="agent")
    plt.title("Cumulative mean regret over time")
    plt.savefig(SAVEDIR + "ex_5_cumulative_{}n_{}bdg_{}prop_{}ex.png".format(MAX_NODES, TOTAL_BUDGET,
                                                                             MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
    plt.show()

