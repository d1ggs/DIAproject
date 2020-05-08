import numpy as np
import os
import time

from social_influence.const import ROOT_PROJECT_PATH, MATRIX_PATH, FEATURE_MAX
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.mc_sampling import MonteCarloSampling
from social_influence.influence_maximisation import GreedyLearner
from social_influence.utils import plot_approx_error_point2

if __name__ == "__main__":
    # Simulate Social Network

    parameters = np.array(
        [[0.1, 0.3, 0.2, 0.2, 0.2], [0.4, 0.1, 0.2, 0.2, 0.1], [0.5, 0.1, 0.1, 0.1, 0.2]])  # parameters for each social

    helper = Helper("facebook_combined")
    dataset = helper.read_dataset("facebook")

    social = SocialNetwork(dataset, parameters[0], FEATURE_MAX)
    prob_matrix = social.get_matrix().copy()
    n_nodes = prob_matrix.shape[0]

    # fake values used for debugging
    # n_nodes = 300
    # prob_matrix = np.random.uniform(0.0,0.01,(n_nodes,n_nodes))

    # mc_sampler = MonteCarloSampling(prob_matrix)

    print("Nodes #: %d" % n_nodes)
    budget = 5
    monte_carlo_simulations = 3
    n_steps_max = 5
    influence_learner = GreedyLearner(prob_matrix, n_nodes)

    start = time.time()
    seeds, influence = influence_learner.parallel_fit(budget, monte_carlo_simulations, n_steps_max)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Best Seeds: [%s] Result: %.2f" % (','.join(str(n) for n in seeds), influence))
    print("Time Elapsed: %d:%d:%d" % (hours, minutes, seconds))

    # plot_name = 'appr_error_n%d_s%d_b%d' % (n_nodes, monte_carlo_simulations, budget)
    # plot_approx_error_point2(sum_simulations, plot_name)
