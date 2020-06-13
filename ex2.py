import numpy as np
import time
import datetime
import argparse

from social_influence.const import FEATURE_MAX, FEATURE_PARAM
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
    parser.add_argument("--plot", action='store_true', help="Plot approximation error")

    parser.add_argument("--mc", default=3, type=int, help="Specify how many mc simulations")
    parser.add_argument("--steps", default=5, type=int, help="Specify how many steps per simulation")
    parser.add_argument("--budget", default=5, type=int, help="Specify budget")
    parser.add_argument("--max_n", default=1000, type=int, help="Specify max number of nodes")
    args = parser.parse_args()

    monte_carlo_simulations = args.mc
    n_steps_max = args.steps
    budget = args.budget
    max_nodes = args.max_n


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
            param = FEATURE_PARAM[0]
            plot_name = plot_name+"_facebook"
        elif args.g:
            dataset = helper.read_dataset("gplus_fixed")
            param = FEATURE_PARAM[1]
            plot_name = plot_name+"_gplus"
        elif args.t:
            dataset = helper.read_dataset("twitter_fixed")
            param = FEATURE_PARAM[2]
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
    if args.plot:
        seeds_max_mc = []
        infl_max_mc = 0
        for i in range(1,monte_carlo_simulations+1):
            seeds, influence = influence_learner.parallel_fit(budget, montecarlo_simulations=i, n_steps_max=n_steps_max)
            results[i] = influence
            print("MC sim: %d Best Seeds: [%s] Result: %.2f" % (i,','.join(str(int(n)) for n in seeds), influence))
            if i == monte_carlo_simulations:
                seeds_max_mc = seeds
                infl_max_mc = influence

        print("Training completed")
        dir_name = "plots/social_influence/nod%d_bud%d_mc%d" % (max_nodes, budget, monte_carlo_simulations)
        plot_approx_error(results,infl_max_mc ,dir_name=dir_name, plot_name=plot_name)
    else:
        seeds, influence = influence_learner.parallel_fit(budget, montecarlo_simulations=monte_carlo_simulations, n_steps_max=n_steps_max)
        print("Training completed")
        print("Best Seeds: [%s] Result: %.2f" % (','.join(str(int(n)) for n in seeds), influence))
    
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time Elapsed: %d:%d:%d" % (hours, minutes, seconds))
