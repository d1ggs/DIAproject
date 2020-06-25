import numpy as np


class LinUCBLearner():
    def __init__(self, feature_matrix, c):
        self.feature_matrix = feature_matrix
        self.c = c
        self.n_nodes = self.feature_matrix.shape[0]
        self.n_features = self.feature_matrix.shape[2]
        self.M = np.identity(self.n_features)
        self.B = np.atleast_2d(np.zeros(self.n_features)).T
        self.collected_rewards = []
        self.pulled_arms = []
        self.theta = []

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
