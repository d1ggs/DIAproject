import numpy as np
from social_influence.IMLinUCB.create_dataset import *
from social_influence.influence_maximisation import *
from social_influence.mc_sampling import *


class IMLinUCBEnviroment():
    def __init__(self, prob_matrix, budget):
        self.prob_matrix = prob_matrix
        self.n_nodes = prob_matrix.shape[0]
        self.budget = budget

    def simulate_episode(self, seeds, n_steps):
        t = 0
        probability_matrix = self.prob_matrix.copy()
        assert (seeds.shape[0] == self.prob_matrix.shape[0])
        history = np.array([seeds])
        active_nodes = seeds
        newly_active_nodes = active_nodes
        all_activated_edges = np.zeros(probability_matrix.shape)
        all_seen_edges = np.zeros(probability_matrix.shape)

        while (t < n_steps-1 and np.sum(newly_active_nodes) > 0):
            p = (probability_matrix.T * active_nodes).T
            activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
            all_activated_edges += activated_edges
            probability_matrix = probability_matrix * ((p != 0) == activated_edges)

            # Activate those nodes which have at least and active edge and were not already active,
            # then add them to the currently active nodes
            newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
            active_nodes = np.array(active_nodes + newly_active_nodes)

            history = np.concatenate((history, [newly_active_nodes]), axis=0)
            t += 1
        for i in range(self.n_nodes):
            if active_nodes[i] != 0:
                all_seen_edges[i, :] = 1
        return history, all_activated_edges, all_seen_edges

    def round(self, seed):
        history, activated_edges, seen_edges = self.simulate_episode(seed, 2)
        n_activated_nodes = np.sum(history)
        #non voglio che in uno stesso round nodo venga attivato più volte
        activated_edges[activated_edges > 0] = 1
        return n_activated_nodes, activated_edges, seen_edges

    def opt(self):
        greedy_learner = GreedyLearner(self.prob_matrix, self.n_nodes)
        best_seed , best_reward = greedy_learner.fit(self.budget, 1000, 2)
        return best_reward
