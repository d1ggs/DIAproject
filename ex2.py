import numpy as np
import time
import datetime
import argparse

from social_influence.const import FEATURE_MAX
from social_influence.helper import Helper
from social_influence.social_setup import SocialNetwork
from social_influence.influence_maximisation import GreedyLearner
from social_influence.utils import plot_approx_error


if __name__ == "__main__":

    # Simulate Social Network
    parser = argparse.ArgumentParser()
    parser.add_argument("-fb", action='store_true', help="Select facebook dataset")
    parser.add_argument("-t", action='store_true', help="Select twitter dataset")
    parser.add_argument("-g", action='store_true', help="Select google dataset")
    parser.add_argument("--test", action='store_true', help="Use random data")

    parser.add_argument("--mc", default=3, type=int, help="Specify how many mc simulations")
    parser.add_argument("--steps", default=5, type=int, help="Specify how many steps per simulation")
    parser.add_argument("--budget", default=5, type=int, help="Specify budget")
    parser.add_argument("--max_n", default=-1, type=int, help="Specify max number of nodes")
    args = parser.parse_args()

    monte_carlo_simulations = args.mc
    n_steps_max = args.steps
    budget = args.budget
    max_nodes = args.max_n

    #TODO include feature weights directly in the dataset?
    parameters = np.asarray(
        ((0.1, 0.3, 0.2, 0.2, 0.2), (0.3, 0.1, 0.2, 0.2, 0.2), (0.5, 0.1, 0.1, 0.1, 0.2)))  # parameters for each social

    helper = Helper()

    plot_name = "appr_error"
    # fake values used for debugging
    if args.test:
        n_nodes = 300
        prob_matrix = np.random.uniform(0.0,0.01,(n_nodes,n_nodes))
        #prob_matrix = np.array([[0 ,0 ,1],[0, 0 ,1 ], [0, 0 ,0]])

        plot_name = plot_name+"_random"
    else:
        if args.fb:
            dataset = helper.read_dataset("facebook_fixed")
            param = parameters[0]
            plot_name = plot_name+"_facebook"
        elif args.g:
            #TODO gplus has node values too high
            dataset = helper.read_dataset("gplus_fixed")
            param = parameters[1]
            plot_name = plot_name+"_gplus"
        elif args.t:
            #TODO twitter has node values too high
            dataset = helper.read_dataset("twitter_fixed")
            param = parameters[2]
            plot_name = plot_name+"_twitter"
        else:
            print("Error: specify which dataset to select. Rerun with --help for info")
            exit(-1)


        social = SocialNetwork(dataset, param, FEATURE_MAX, max_nodes=max_nodes)
        prob_matrix = social.get_matrix().copy()
        n_nodes = social.get_n_nodes()
        
    print("Nodes #: %d" % n_nodes)
    influence_learner = GreedyLearner(prob_matrix, n_nodes)

    start = time.time()
    
    results = {}
    # Plot the approximation error as the parameters of the algorithms vary for every specific network.
    for i in range(1,monte_carlo_simulations+1):
        seeds, influence = influence_learner.parallel_fit(budget, montecarlo_simulations=i, n_steps_max=n_steps_max)
        results[i] = influence

        if i == monte_carlo_simulations:
            seeds_max_mc = seeds
            infl_max_mc = influence
    
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training completed")
    print("Time Elapsed: %d:%d:%d" % (hours, minutes, seconds))


    plot_approx_error(results,infl_max_mc ,plot_name=plot_name)

    print("Best Seeds: [%s] Result: %.2f" % (','.join(str(int(n)) for n in seeds_max_mc), infl_max_mc))