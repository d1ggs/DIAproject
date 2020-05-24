import copy
import json
from multiprocessing import Pool

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tqdm import trange, tqdm

from pricing.conversion_rate import ProductConversionRate
from pricing.environments import NonStationaryEnvironment
from pricing.learners.UCBLearner import  SWUCBLearner
from pricing.learners.swts_learner import SWTSLearner
from pricing.const import *
from pricing.learners.ts_learner import TSLearner
from social_influence.const import ROOT_PROJECT_PATH, MATRIX_PATH, FEATURE_MAX
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.influence_maximisation import GreedyLearner

# Social influence constants
MAX_NODES = 600
TOTAL_BUDGET = 100

SOCIAL_NAMES = ["gplus", "facebook", "twitter"]
PARAMETERS = np.array(
    [[0.1, 0.3, 0.2, 0.2, 0.2],
     [0.4, 0.1, 0.2, 0.2, 0.1],
     [0.5, 0.1, 0.1, 0.1, 0.2]])  # parameters for each social

def run_experiment(initial_seeds, n_arms, prices, horizon, conversion_curves, sampler):
    # Restore the seeds
    seeds = np.copy(initial_seeds)
    # Reset the environments
    ts_env = NonStationaryEnvironment(curves=conversion_curves, horizon=horizon, prices=prices)
    ts_learner = TSLearner(prices=prices)
    swts_env = NonStationaryEnvironment(curves=conversion_curves, horizon=horizon, prices=prices)
    swts_learner = SWTSLearner(prices=prices, horizon=horizon, const=60)
    swucb_env = NonStationaryEnvironment(curves=conversion_curves, horizon=horizon, prices=prices)
    swucb_learner = SWUCBLearner(n_arms=n_arms, horizon=horizon, prices=prices, const=60)
    regrets_swts_per_timestep = []
    regrets_swucb_per_timestep = []
    regrets_ts_per_timestep = []
    cumulative_regret_swts = cumulative_regret_swucb = cumulative_regret_ts = 0
    tot = 0
    # Pricing loop
    for i in range(horizon):
        # Advance the propagation in the social network
        seeds_vector = sampler.simulate_episode(seeds, 1)

        if seeds_vector.shape[0] == 1:  # The propagation has stopped, no need to continue the loop
            break

        seeds = seeds_vector[1]
        clicks = int(np.sum(seeds))

        # Bandit pricing

        opt_reward = ts_env.opt_reward()

        tot += clicks

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

    return regrets_swucb_per_timestep, regrets_swts_per_timestep, regrets_ts_per_timestep


if __name__ == "__main__":

    # TODO divide budget among social networks

    budget_1, budget_2, budget_3 = 10, 10, 10

    # Simulate Social Network

    for social_network, product_index in zip(SOCIAL_NAMES, range(len(SOCIAL_NAMES))):
        helper = Helper(social_network + "_combined")
        dataset = helper.read_dataset(social_network + "_fixed")

        social = SocialNetwork(dataset, PARAMETERS[product_index], FEATURE_MAX, max_nodes=MAX_NODES)
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

        curves = []

        # Load product conversion rate curve information
        with open("pricing/products/products.json", 'r') as productfile:
            p_info = json.load(productfile)
            productfile.close()

            product = p_info["products"][product_index]
            product_id = product["product_id"]
            for season in product["seasons"]:
                season_id = season["season_id"]
                y = season["y_values"]
                curves.append(ProductConversionRate(product_id, season_id, N_ARMS, y))

        # Support variables to store results for each experiment
        swucb_regrets_per_experiment = []
        swts_regrets_per_experiment = []
        ts_regrets_per_experiment = []

        # Store the original seeds to repeat experiments
        original_seeds = np.copy(seeds)

        # Run multiple pricing experiments in parallel

        results_async = []
        results = []

        with Pool() as pool:
            for _ in range(N_EXPERIMENTS):
                r_async = pool.apply_async(run_experiment,
                                           args=(original_seeds, N_ARMS, PRICES, TIME_HORIZON, curves,
                                                 copy.deepcopy(mc_sampler)))
                results_async.append(r_async)

            for r in tqdm(results_async):
                results.append(r.get())

        for regrets_swucb_per_timestep, regrets_swts_per_timestep, regrets_ts_per_timestep in results:
            swucb_regrets_per_experiment.append(regrets_swucb_per_timestep)
            swts_regrets_per_experiment.append(regrets_swts_per_timestep)
            ts_regrets_per_experiment.append(regrets_ts_per_timestep)

            # print("Total pulls:", tot)

        # Plot results

        agents = ["SW-UCB", "SW-TS", "TS"]
        regrets = [swucb_regrets_per_experiment, swts_regrets_per_experiment, ts_regrets_per_experiment]

        labels = []
        results = []
        timesteps = []
        indexes = []

        # Prepare the data structures for the dataframe
        for agent, data in zip(agents, regrets):
            for experiment, index in zip(data, range(len(data))):
                labels.extend([agent] * len(experiment))
                timesteps.extend(np.arange(len(experiment)))
                results.extend(experiment)
                indexes.extend([index] * len(experiment))

        plt.figure()
        df = pd.DataFrame({"agent": labels, "regret": results, "timestep": timesteps, "experiment_id": indexes})
        sns.lineplot(x="timestep", y="regret", data=df, hue="agent")
        plt.title(social_network + " - product " + str(product_index + 1) + " : mean regret over time")
        plt.show()