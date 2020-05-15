import json

import numpy as np
import os
import time
import matplotlib.pyplot as plt

from tqdm import trange

from pricing.conversion_rate import ProductConversionRate
from pricing.environments import StationaryEnvironment
from pricing.learners.UCBLearner import UCBLearner
from pricing.learners.ts_learner import TSLearner
from pricing.const import *
from social_influence.const import ROOT_PROJECT_PATH, MATRIX_PATH, FEATURE_MAX
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.influence_maximisation import GreedyLearner

# Social influence constants
MAX_NODES = 300
TOTAL_BUDGET = 100

social_names = ["facebook", "gplus", "twitter"]
parameters = np.array(
        [[0.1, 0.3, 0.2, 0.2, 0.2],
         [0.4, 0.1, 0.2, 0.2, 0.1],
         [0.5, 0.1, 0.1, 0.1, 0.2]])  # parameters for each social

if __name__ == "__main__":

    # TODO divide budget among social networks

    budget_1, budget_2, budget_3 = 10, 10, 10

    # Simulate Social Network

    for social_network, index in zip(social_names, range(len(social_names))):

        helper = Helper(social_network + "_combined")
        dataset = helper.read_dataset(social_network)

        social = SocialNetwork(dataset, parameters[index], FEATURE_MAX, max_nodes=MAX_NODES)
        prob_matrix = social.get_matrix().copy()
        n_nodes = prob_matrix.shape[0]

        # fake values used for debugging
        # n_nodes = 300
        # prob_matrix = np.random.uniform(0.0,0.01,(n_nodes,n_nodes))

        mc_sampler = MonteCarloSampling(prob_matrix)

        print("Nodes #: %d" % n_nodes)
        monte_carlo_simulations = 3
        n_steps_max = 5
        influence_learner = GreedyLearner(prob_matrix, n_nodes)

        start = time.time()
        seeds, influence = influence_learner.parallel_fit(budget_1, monte_carlo_simulations, n_steps_max)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Best Seeds: [%s] Result: %.2f" % (','.join(str(int(n)) for n in seeds), influence))
        print("Time Elapsed: %d:%d:%d" % (hours, minutes, seconds))

        print("Activated seeds:", np.sum(seeds))


        # Load product conversion rate curve information
        with open("pricing/products/products.json", 'r') as productfile:
            p_info = json.load(productfile)
            productfile.close()

            product = p_info["products"][index]
            product_id = product["product_id"]
            seasons = product["seasons"]
            season_id = seasons[0]["season_id"]
            y = seasons[0]["y_values"]
            curve = ProductConversionRate(product_id, season_id, N_ARMS, y)

        # Support variables
        env = StationaryEnvironment(prices=PRICES, curve=curve)

        # ucb_learner = UCBLearner(N_ARMS, PRICES)
        ts_learner = TSLearner(N_ARMS, PRICES)

        # cumulative_regret_ucb = 0
        cumulative_regret_ts = 0
        # regrets_ucb_per_timestep = []
        regrets_ts_per_timestep = []

        # Pricing loop
        for i in trange(TIME_HORIZON):
            # Advance the propagation in the social network
            seeds_vector = mc_sampler.simulate_episode(seeds, 1)

            if seeds_vector.shape[0] == 1: # The propagation has stopped, no need to continue the loop
                break

            seeds = seeds_vector[1]
            clicks = int(np.sum(seeds))

            # Bandit pricing

            # Choose a price for each user and compute reward
            for _ in range(clicks):
                # UCB learner
                # pulled_arm = ucb_learner.pull_arm()
                # reward = env.round(pulled_arm)
                # ucb_learner.update(pulled_arm, reward)
                #
                # instantaneous_regret = env.get_inst_regret(pulled_arm)
                # cumulative_regret_ucb += instantaneous_regret

                # TS learner
                pulled_arm = ts_learner.pull_arm()
                reward = env.round(pulled_arm)
                ts_learner.update(pulled_arm, reward)

                instantaneous_regret = env.get_inst_regret(pulled_arm)
                cumulative_regret_ts += instantaneous_regret

            # regrets_ucb_per_timestep.append(cumulative_regret_ucb)
            regrets_ts_per_timestep.append(cumulative_regret_ts)


        # Plot the regret over time
        plt.figure()
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.title(social_network + " - product " + str(index + 1))
        # plt.plot(regrets_ucb_per_timestep, 'r')
        plt.plot(regrets_ts_per_timestep, 'b')
        # plt.legend(['UCB1', "TS"])
        plt.show()

        # plot_name = 'appr_error_n%d_s%d_b%d' % (n_nodes, monte_carlo_simulations, budget)
        # plot_approx_error_point2(sum_simulations, plot_name)
