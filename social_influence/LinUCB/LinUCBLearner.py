import numpy as np
from social_influence.influence_maximisation import GreedyLearner

class LinUCBLearner():
    def __init__(self, feature_matrix, mc_simulations, n_steps, budget, c=2):
        self.feature_matrix = feature_matrix
        self.c = c
        self.n_nodes = self.feature_matrix.shape[0]
        self.n_features = self.feature_matrix.shape[2]
        self.M = np.identity(self.n_features)
        self.B = np.atleast_2d(np.zeros(self.n_features)).T
        self.collected_rewards = []
        self.pulled_arms = []
        self.theta = []

        self.theta = np.atleast_2d(np.zeros(self.feature_matrix.shape[2]))
        self.n_experiment = 0
        self.prob_matrix = np.zeros((self.feature_matrix.shape[0], self.feature_matrix.shape[0]))
        self.budget = budget
        self.mc_simulations = mc_simulations
        self.n_steps = n_steps

    def compute_ucbs(self):
        self.theta = np.dot(np.linalg.inv(self.M), self.B)
        ucbs = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_features = np.atleast_2d(self.feature_matrix[i, j]).T
                ucbs[i, j] = np.dot(edge_features.T, self.theta) + self.c * np.sqrt(
                    np.dot(edge_features.T, np.dot(np.linalg.inv(self.M), edge_features)))
        return ucbs

    def get_theta(self):
        return self.theta

    def pull_arm(self):
        ucbs = self.compute_ucbs()
        pulled_arm = np.unravel_index(np.argmax(ucbs), ucbs.shape)
        return pulled_arm

    def update_values(self, arm_index, reward):
        self.collected_rewards.append(reward)
        self.pulled_arms.append(arm_index)
        pulled_edge_features = np.atleast_2d(self.feature_matrix[arm_index[0], arm_index[1]]).T
        self.M += np.dot(pulled_edge_features, pulled_edge_features.T)
        self.B += pulled_edge_features * reward

#    def update_tetha(self,extimated_theta):
#        self.theta += extimated_theta.T

    def __calc_prob_matrix(self):
        for i in range(self.feature_matrix.shape[0]):
            for j in range(self.feature_matrix.shape[0]):
                self.prob_matrix[i,j] = np.dot(self.feature_matrix[i,j,:],self.theta.T)

    def get_prob_matrix(self):
        self.__calc_prob_matrix()
        return self.prob_matrix

    def find_best_seeds(self, parallel = True):
        self.__calc_prob_matrix()
        greedy_learner = GreedyLearner(self.prob_matrix, self.feature_matrix.shape[0])
        if parallel:
            seed, reward = greedy_learner.parallel_fit(self.budget,self.mc_simulations,self.n_steps)
        else:
            seed, reward = greedy_learner.fit(self.budget, self.mc_simulations, self.n_steps)
        return seed, reward

    # def get_final_matrix(self,n_experiments):
    #     final_theta = self.theta/self.n_experiment
    #     for i in range(self.feature_matrix.shape[0]):
    #         for j in range(self.feature_matrix.shape[0]):
    #             self.prob_matrix[i,j] = np.dot(self.feature_matrix[i,j,:],final_theta.T)