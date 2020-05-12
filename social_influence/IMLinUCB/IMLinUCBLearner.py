from social_influence.influence_maximisation import *
from social_influence.mc_sampling import *


class IMLinUCBLearner():
    def __init__(self, n_features, feature_matrix_edges, budget):
        self.M = np.eye(n_features)
        self.B = np.atleast_2d(np.zeros(n_features)).T
        self.collected_rewards = []
        self.feature_matrix_edges = feature_matrix_edges
        self.sigma = 1
        self.c = 2
        self.budget = budget
        self.n_nodes = feature_matrix_edges.shape[0]

    def pull_arm(self):
        UCB_matrix = np.zeros((self.n_nodes, self.n_nodes))
        theta = self.sigma ** (-2) * np.dot(np.linalg.inv(self.M), self.B)
        max_ucb = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                arm = np.atleast_2d(self.feature_matrix_edges[i, j]).T
                ucb = np.dot(theta.T, arm) + self.c * np.sqrt(np.dot(arm.T, np.dot(np.linalg.inv(self.M), arm)))
                if ucb[0, 0] > max_ucb:
                   max_ucb = ucb[0, 0]
                UCB_matrix[i, j] = ucb[0, 0]
                if UCB_matrix[i,j] <0:
                    UCB_matrix[i,j] = 0
        UCB_matrix = UCB_matrix / max_ucb
        print(UCB_matrix)
        oracolo = GreedyLearner(UCB_matrix, self.n_nodes)
        pulled_arm, _ = oracolo.fit(self.budget, 3, 3)
        return pulled_arm

    def update_observations(self, reward, activated_edges):
        self.collected_rewards.append(reward)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if activated_edges[i, j] > 0:
                    arm = np.atleast_2d(self.feature_matrix_edges[i, j].T)
                    self.M = self.M + self.sigma ** (-2) * np.dot(arm.T, arm)
                    self.B = self.B + arm.T
