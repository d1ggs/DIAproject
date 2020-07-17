import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange
import copy

from pricing.conversion_rate import ProductConversionRate
from pricing.environments import NonStationaryEnvironment
from pricing.learners.UCBLearner import SWUCBLearner
from pricing.learners.ts_learner import TSLearner
from pricing.learners.swts_learner import SWTSLearner
from pricing.const import *
from social_influence.const import FEATURE_MAX, FEATURE_PARAM, SOCIAL_NAMES, MAX_NODES
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.budget_allocation import CumulativeBudgetAllocation

# Overwritten constants
TOTAL_BUDGET = 5
MAX_PROPAGATION_STEPS = 3
N_EXPERIMENTS = 5
TIME_HORIZON = 360
monte_carlo_simulations = 3
n_steps_max = 5

SAVEDIR = "./plots/ex_6/"

if __name__ == "__main__":

    social_networks = []
    products = []

    # Load the products information
    with open(PRODUCT_FILE, 'r') as productfile:
        p_info = json.load(productfile)
        productfile.close()

    for product_index in range(3):
        product = p_info["products"][product_index]
        products.append(product)

    print("Loading social networks and products...\n")

    # Social network generation
    for social_network, index in zip(SOCIAL_NAMES, range(len(SOCIAL_NAMES))):
        helper = Helper()
        dataset = helper.read_dataset(social_network + "_fixed")

        # Each social network as a number of nodes equal to MAX_NODES
        social = SocialNetwork(dataset, FEATURE_PARAM[index], FEATURE_MAX, max_nodes=MAX_NODES)
        social_networks.append(social)

    ts_regrets_per_experiment = [[], [], []]
    swucb_regrets_per_experiment = [[], [], []]
    swts_regrets_per_experiment = [[], [], []]

    print("\nPrecomputing social influence for maximum budget...")

    budget_allocator = CumulativeBudgetAllocation(social_networks[0].get_matrix(), social_networks[1].get_matrix(),
                                                  social_networks[2].get_matrix(), TOTAL_BUDGET,
                                                  monte_carlo_simulations, MAX_PROPAGATION_STEPS)

    print("\nPerforming experiments...")

    original_envs = []
    original_ts_learners = []
    original_swucb_learners = []
    original_swts_learners = []

    # Conversion rate curves generation, Non-stationary environments and Learners initial setting
    for i in range(3):
        product = products[i]
        product_id = product["product_id"]
        seasons = product["seasons"]
        curves = []

        # Three phases for each product
        for season in seasons:
            y = season["y_values"]
            season_id = season["season_id"]
            curves.append(ProductConversionRate(product_id, season_id, N_ARMS, y))
        original_envs.append(NonStationaryEnvironment(prices=PRICES, curves=curves, horizon=TIME_HORIZON))

        original_ts_learners.append(TSLearner(PRICES))
        original_swucb_learners.append(SWUCBLearner(TIME_HORIZON, PRICES, const=SLIDING_WINDOW_CONSTANTS[i]))
        original_swts_learners.append(SWTSLearner(PRICES, TIME_HORIZON, const=SLIDING_WINDOW_CONSTANTS[i]))

    for _ in range(N_EXPERIMENTS):

        # Initialization of the regrets
        cumulative_regret_ts = [0, 0, 0]
        cumulative_regret_swucb = [0, 0, 0]
        cumulative_regret_swts = [0, 0, 0]

        regrets_ts_per_timestep = [[], [], []]
        regrets_swucb_per_timestep = [[], [], []]
        regrets_swts_per_timestep = [[], [], []]

        # Generation of new environment and learners for the current experiment
        envs = copy.deepcopy(original_envs)
        swucb_learners = copy.deepcopy(original_swucb_learners)
        ts_learners = copy.deepcopy(original_ts_learners)
        swts_learners = copy.deepcopy(original_swts_learners)

        for _ in trange(TIME_HORIZON):

            # When a new phase starts, a new optimal arm is calculated
            if envs[0].new_phase():

                # Compute the optimal prices for each product for the current phase
                opt_weights = []
                best_arms = []
                for j in range(3):
                    weight, best_arm = envs[j].opt_reward()
                    opt_weights.append(weight)
                    best_arms.append(best_arm)

                # Run the joint influence maximisation to find the optimal number of clicks with the current
                # optimal price
                budget, average_joint_influence, best_seeds = budget_allocator.joint_influence_maximization(
                    weights=opt_weights,
                    split_joint_influence=True)

                opt_clicks = [int(round(average_joint_influence[w])) for w in range(3)]

            # Compute the joint inflence maximisation using the current best prices as weights for each social network
            weights = [swts_learners[i].get_last_best_price() for i in range(3)]
            budget, joint_influence, seeds = budget_allocator.joint_influence_maximization(weights=weights,
                                                                                           split_joint_influence=True)
            for i in range(3):
                opt_clairvoyant = opt_clicks[i] * opt_weights[i]

                # The number of clicks for the timestep is given by the social influence value = number of activated nodes
                learner_clicks = int(round(joint_influence[i]))

                # Bandit pricing
                reward_swucb = 0
                reward_ts = 0
                reward_swts = 0
                reward_clairvoyant = 0

                # Choose a price for each user and compute the reward
                for _ in range(learner_clicks):
                    # SW-UCB learner
                    pulled_arm = swucb_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    swucb_learners[i].update(pulled_arm, reward)
                    reward_swucb += reward * PRICES[pulled_arm]

                    # TS learner
                    pulled_arm = ts_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    ts_learners[i].update(pulled_arm, reward)
                    reward_ts += reward * PRICES[pulled_arm]

                    # SW-TS learner
                    pulled_arm = swts_learners[i].pull_arm()
                    reward = envs[i].round(pulled_arm)
                    swts_learners[i].update(pulled_arm, reward)
                    reward_swts += reward * PRICES[pulled_arm]

                # Compute the reward given the optimal arm and optimal number of clicks
                for _ in range(opt_clicks[i]):
                    reward_clairvoyant += envs[i].round(best_arms[i]) * PRICES[best_arms[i]]

                # Advance time in the environment
                envs[i].forward_time()

                # Calculate and append the cumulative regret of the timestep
                cumulative_regret_swucb[i] += reward_clairvoyant - reward_swucb
                cumulative_regret_ts[i] += reward_clairvoyant - reward_ts
                cumulative_regret_swts[i] += reward_clairvoyant - reward_swts

                regrets_ts_per_timestep[i].append(cumulative_regret_ts[i])
                regrets_swucb_per_timestep[i].append(cumulative_regret_swucb[i])
                regrets_swts_per_timestep[i].append(cumulative_regret_swts[i])

        for i in range(3):
            ts_regrets_per_experiment[i].append(regrets_ts_per_timestep[i])
            swucb_regrets_per_experiment[i].append(regrets_swucb_per_timestep[i])
            swts_regrets_per_experiment[i].append(regrets_swts_per_timestep[i])

    # Plot results

    agents = ["TS", "SW-UCB", "SW-TS"]
    regrets = [ts_regrets_per_experiment, swucb_regrets_per_experiment, swts_regrets_per_experiment]

    labels = []
    results = []
    timesteps = []
    indexes = []

    print("Saving plots...")

    for social_network, product_index, ts_regret, swucb_regret, swts_regret in zip(SOCIAL_NAMES, range(3),
                                                                                   ts_regrets_per_experiment,
                                                                                   swucb_regrets_per_experiment,
                                                                                   swts_regrets_per_experiment):
        for agent, data in zip(agents, [ts_regret, swucb_regret, swts_regret]):

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
        plt.savefig(SAVEDIR + "ex_6_{}_{}n_{}bdg_{}prop_{}ex.png".format(social_network, MAX_NODES, TOTAL_BUDGET,
                                                                         MAX_PROPAGATION_STEPS, N_EXPERIMENTS))
        plt.show()
