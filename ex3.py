import numpy as np
import time
import argparse

from social_influence.const import FEATURE_MAX
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.utils import plot_approx_error
from social_influence.budget_allocation import GreedyBudgetAllocation

if __name__ == "__main__":

    # Simulate Social Network
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help="Use random data")

    parser.add_argument("--mc", default=3, type=int, help="Specify how many mc simulations")
    parser.add_argument("--steps", default=5, type=int, help="Specify how many steps per simulation")
    parser.add_argument("--budget", default=5, type=int, help="Specify budget")
    args = parser.parse_args()

    monte_carlo_simulations = args.mc
    n_steps_max = args.steps
    budget = args.budget

    # TODO include feature weights directly in the dataset?
    parameters = np.asarray(
        ((0.1, 0.3, 0.2, 0.2, 0.2), (0.3, 0.1, 0.2, 0.2, 0.2), (0.5, 0.1, 0.1, 0.1, 0.2)))  # parameters for each social

    helper = Helper()

    plot_name = "appr_error"
    max_node = -1
    # fake values used for debugging
    if args.test:
        max_node = 300

    print("Initializing Social Networks...")

    facebook = helper.read_dataset("facebook_fixed", )
    social1 = SocialNetwork(facebook, parameters[0], FEATURE_MAX, max_nodes=max_node)

    gplus = helper.read_dataset("gplus_fixed")
    social2 = SocialNetwork(gplus, parameters[1], FEATURE_MAX, max_nodes=max_node)

    twitter = helper.read_dataset("twitter_fixed")
    social3 = SocialNetwork(twitter, parameters[2], FEATURE_MAX, max_nodes=max_node)

    budget_allocator = GreedyBudgetAllocation(social1, social2, social3, budget, monte_carlo_simulations, n_steps_max)

    start = time.time()

    print("Start Budget Allocation..")
    # budget_allocator.joint_influence_maximization()
    budget_allocator.test_join_influence()

    # # Plot the approximation error as the parameters of the algorithms vary for every specific network.
    # results = {}
    # for i in range(1,monte_carlo_simulations+1):
    #     seeds, influence = influence_learner.parallel_fit(budget, montecarlo_simulations=i, n_steps_max=n_steps_max)
    #     results[i] = influence

    #     if i == monte_carlo_simulations:
    #         seeds_max_mc = seeds
    #         infl_max_mc = influence

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training completed")
    print("Time Elapsed: %d:%d:%d" % (hours, minutes, seconds))

    # plot_approx_error(results,infl_max_mc ,plot_name=plot_name)

    # print("Best Seeds: [%s] Result: %.2f" % (','.join(str(int(n)) for n in seeds_max_mc), infl_max_mc))
