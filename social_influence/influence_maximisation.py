import numpy as np
from itertools import combinations
import multiprocessing
from multiprocessing import Process, Queue, Pool
from abc import ABC, abstractmethod
from tqdm import tqdm

from social_influence.mc_sampling import MonteCarloSampling


class SingleInfluenceLearner(ABC):
    """
    Abstract Class that represent a Influence Maximisation learner for a single Social Network

    Attributes
    --------
    prob_matrix : Edge Activation Probabilities matrix

    n_nodes : number of nodes in the graph

    """

    def __init__(self, prob_matrix, n_nodes: int):
        self.prob_matrix = prob_matrix
        self.n_nodes = n_nodes

    @abstractmethod
    def fit(self):
        pass


class GreedyLearner(SingleInfluenceLearner):

    def pool_worker(self, node: int, best_seeds: np.ndarray, mc_sim: int, n_steps: int):
        """
        Function that runs in a single thread. Used only for parallel_fit

        Parameters
        -----------
        node : node added to the seeds at this step to compute influence

        best_seeds : array of seeds computed at previous steps of the Greedy Algorithm

        mc_sim : montecarlo_simulation : number of MonteCarlo Simulations

        n_steps_max : max number of steps in a simulation
        """
        sampler = MonteCarloSampling(self.prob_matrix)
        seeds = np.copy(best_seeds)
        seeds[node] = 1  # add current node to seeds

        nodes_probabilities = sampler.mc_sampling(seeds, mc_sim, n_steps)
        influence = np.sum(nodes_probabilities)

        return node, influence

    def parallel_fit(self, budget: int, montecarlo_simulations: int, n_steps_max: int):
        """
        Greedy influence maximization algorithm. Execution in parallel
        
        At each step, a pool of workers computes the influence of the best seeds of the previous step plus a new node.
        The node with the best marginal increase is added to the best seeds.
        The number of iterations is given by the budget

        Parameters
        ---------
        budget : budget for this social network

        montecarlo_simulation : number of MonteCarlo Simulations

        n_steps_max : max number of steps in a simulation
        """

        max_influence = 0
        best_seeds = np.zeros(self.n_nodes)
        
        print("Start Greedy Influence Maximisation with budget: %d, mc_simulations: %d, n_steps_max: %d" % (budget, montecarlo_simulations, n_steps_max))
        for i in range(budget):
            #best_marginal_increase = 0 
            step_influence = 0

            results_async = []
            results = []
            with Pool() as pool:
                for n in range(self.n_nodes):
                    if best_seeds[n] == 0:
                        r_async = pool.apply_async(self.pool_worker,
                                                args=(n, best_seeds, montecarlo_simulations, n_steps_max))
                        results_async.append(r_async)

                for r in tqdm(results_async):
                    results.append(r.get())

                pool.close()

            for res in results:
                n, influence = res
                #marginal_increase = influence - step_influence
                
                if influence >= step_influence:
                    best_node = n
                    #best_marginal_increase = marginal_increase
                    step_influence = influence


            best_seeds[best_node] = 1
            max_influence = step_influence
            print("Node with best marginal increase at step %d: %d" % (i + 1, best_node))

        print('-'*100)
        return best_seeds, max_influence

    def fit(self, budget: int, montecarlo_simulations: int, n_steps_max: int):
        """
        Greedy influence maximization algorithm. Serial execution
        
        Parameters
        ---------
        budget : budget for this social network

        montecarlo_simulation : number of MonteCarlo Simulations

        n_steps_max : max number of steps in a simulation
        """

        sampler = MonteCarloSampling(self.prob_matrix)

        max_influence = 0
        best_seeds = np.zeros(self.n_nodes)
        for i in range(budget):
            #best_marginal_increase = 0
            step_influence = 0

            for n in range(self.n_nodes):
                if best_seeds[n] == 0:
                    # computer marginal increase
                    seeds = np.copy(best_seeds)
                    seeds[n] = 1  # add current node to seeds

                    # computer marginal increase
                    nodes_probabilities = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)
                    influence = np.sum(nodes_probabilities)

                    #marginal_increase = influence - step_influence

                    if influence >= step_influence:
                        best_node = n
                        #best_marginal_increase = marginal_increase
                        step_influence = influence
                if n % 100 == 0:
                    print("Analysing node: %d of %d" % (n, self.n_nodes))

            best_seeds[best_node] = 1
            max_influence = step_influence
            print("Node with best marginal increase at step %d: %d" % (i + 1, best_node))

        return best_seeds, max_influence

    def cumulative_fit(self, budget: int, montecarlo_simulations: int, n_steps_max: int):
        """
        Greedy influence maximization algorithm. Serial execution. Returns an array with tuple (node_step_i , reward_step_i)
        
        Parameters
        ---------
        budget : budget for this social network

        montecarlo_simulation : number of MonteCarlo Simulations

        n_steps_max : max number of steps in a simulation
        """

        sampler = MonteCarloSampling(self.prob_matrix)

        #max_influence = 0
        best_seeds = np.zeros(self.n_nodes)
        results = []
        for i in range(budget):
            #best_marginal_increase = 0
            step_influence = 0

            for n in tqdm(range(self.n_nodes)):
                if best_seeds[n] == 0:
                    # computer marginal increase
                    seeds = np.copy(best_seeds)
                    seeds[n] = 1  # add current node to seeds

                    # computer marginal increase
                    nodes_probabilities = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)
                    influence = np.sum(nodes_probabilities)

                    #marginal_increase = influence - step_influence

                    if influence >= step_influence:
                        best_node = n
                        #best_marginal_increase = marginal_increase
                        step_influence = influence
                # if n % 100 == 0:
                #     print("Analysing node: %d of %d" % (n, self.n_nodes))

            best_seeds[best_node] = 1
            #max_influence = step_influence
            results.append((best_node, step_influence))
            print("Node with best marginal increase at step %d: %d" % (i + 1, best_node))

        return results

    def cumulative_parallel_fit(self, budget: int, montecarlo_simulations: int, n_steps_max: int):
            """
            Greedy influence maximization algorithm. Execution in parallel. Returns an array with tuple (node_step_i , reward_step_i)
            
            At each step, a pool of workers computes the influence of the best seeds of the previous step plus a new node.
            The node with the best marginal increase is added to the best seeds.
            The number of iterations is given by the budget

            Parameters
            ---------
            budget : budget for this social network

            montecarlo_simulation : number of MonteCarlo Simulations

            n_steps_max : max number of steps in a simulation
            """

            max_influence = 0
            best_seeds = np.zeros(self.n_nodes)
            
            print("Start Greedy Influence Maximisation with budget: %d, mc_simulations: %d, n_steps_max: %d" % (budget, montecarlo_simulations, n_steps_max))
            cumulative_results = []
            for i in range(budget):
                #best_marginal_increase = 0 
                step_influence = 0

                results_async = []
                results = []
                with Pool() as pool:
                    for n in range(self.n_nodes):
                        if best_seeds[n] == 0:
                            r_async = pool.apply_async(self.pool_worker,
                                                    args=(n, best_seeds, montecarlo_simulations, n_steps_max))
                            results_async.append(r_async)

                    for r in tqdm(results_async):
                        results.append(r.get())

                    pool.close()

                for res in results:
                    n, influence = res
                    #marginal_increase = influence - step_influence
                    
                    if influence >= step_influence:
                        best_node = n
                        #best_marginal_increase = marginal_increase
                        step_influence = influence


                best_seeds[best_node] = 1
                max_influence = step_influence
                cumulative_results.append((best_node, step_influence))
                print("Node with best marginal increase at step %d: %d" % (i + 1, best_node))

            print('-'*100)
            return cumulative_results
        
    def step_fit(self, montecarlo_simulations: int, n_steps_max: int, resume_seeds=None):
        """
        Greedy influence maximization algorithm. Serial execution. Returns an array with tuple (node_step_i , reward_step_i)
        
        Parameters
        ---------

        montecarlo_simulation : number of MonteCarlo Simulations

        n_steps_max : max number of steps in a simulation
        """

        sampler = MonteCarloSampling(self.prob_matrix)

        best_seeds = np.zeros(self.n_nodes)
        if resume_seeds:
            best_seeds[resume_seeds] = 1

        step_influence = 0

        for n in tqdm(range(self.n_nodes)):
            if best_seeds[n] == 0:
                # computer marginal increase
                seeds = np.copy(best_seeds)
                seeds[n] = 1  # add current node to seeds

                # computer marginal increase
                nodes_probabilities = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)
                influence = np.sum(nodes_probabilities)

                #marginal_increase = influence - step_influence

                if influence >= step_influence:
                    best_node = n
                    #best_marginal_increase = marginal_increase
                    step_influence = influence
            # if n % 100 == 0:
            #     print("Analysing node: %d of %d" % (n, self.n_nodes))

        best_seeds[best_node] = 1
            #max_influence = step_influence

        return (best_node, step_influence)


class ExactSolutionLearner(SingleInfluenceLearner):
    """
    Exact Solution Influence Maximisation
    """

    def fit(self, budget: int, montecarlo_simulations: int, n_steps_max: int):
        """
        Basic exact influence maximization algorithm which enumerates all seeds node given a budget. Returns indexes of best seeds 
        
        Parameters
        ---------
        budget : budget for this social network

        montecarlo_simulation : number of MonteCarlo Simulations

        n_steps_max : max number of steps in a simulation
        """

        sampler = MonteCarloSampling(self.prob_matrix)

        nodes = [d for d in range(0, self.n_nodes)]
        # seeds_combinations = list(combinations(nodes, self.budget)) #this take too much memory

        max_influence = 0
        best_seeds = np.zeros(self.n_nodes)

        for combination in combinations(nodes, budget):  # enumerate all possible seeds given a budget
            seeds = np.zeros(self.n_nodes)
            # print("Current Seeds: [%s]" % (','.join(str(n) for n in combination)))
            seeds[list(combination)] = 1

            nodes_probabilities = sampler.mc_sampling(seeds, montecarlo_simulations, n_steps_max)

            # the best seed is the one where the sum of probabilities is the highest
            influence = np.sum(nodes_probabilities)

            if (influence >= max_influence):
                max_influence = influence
                best_seeds = seeds

        # print("Best Seeds: [%s] Result: %.2f" % (','.join(str(n) for n in seeds), max_influence))

        return best_seeds, max_influence
