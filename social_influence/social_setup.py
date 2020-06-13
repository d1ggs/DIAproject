import math

import numpy as np
import pandas as pd
import os
import math



class SocialNetwork:
    def __init__(self, dataset, parameters, feature_max, max_nodes=-1):
        """
        Features in a Social Network are: Tag, Share, Like, Message, Comment
        They are saved in self.features as a numpy array, ordered as written above (self.features[0] -> Tag...)
        """

        self.social_edges, self.features = dataset
        self.parameters = parameters
        self.feature_max = feature_max
        self.max_nodes = max_nodes

        assert(self.parameters.shape == self.features[0].shape)

        # Use math.isclose because np.sum may don't return exactly 1
        assert (math.isclose(np.sum(self.parameters), 1.0, rel_tol=1e-5))

        self.matrix = self.probability_matrix()

    def compute_activation_prob(self, features):
        out = np.dot(self.parameters, features)  # dot product
        prob = out / self.feature_max  # divide by the maximum value of a feature
        return prob

    def probability_matrix(self):
        max_node = self.social_edges.max()
        if self.max_nodes > 0:
            max_node = min([self.max_nodes, max_node])
            print("Reducing dataset matrix to {} x {} nodes".format(max_node, max_node))

        matrix = np.zeros((max_node, max_node))

        for i in range(self.social_edges.shape[0]):
            node_a = self.social_edges[i][0]
            node_b = self.social_edges[i][1]

            # Write the activation probability only if the nodes are in range
            if node_a < max_node and node_b < max_node:
                features = self.features[i]
                matrix[node_a, node_b] = self.compute_activation_prob(features)

        return matrix

    def get_matrix(self):
        return self.matrix

    def get_n_nodes(self):
        return self.matrix.shape[0]

    def get_edge_features_matrix(self):
        max_node = self.social_edges.max()
        if self.max_nodes > 0:
            max_node = min([self.max_nodes, max_node])
            print("Reducing dataset matrix to {} x {} nodes".format(max_node, max_node))

        feature_size = self.features.shape[1]
        matrix = np.zeros(shape=(max_node, max_node, feature_size))

        for i in range(self.social_edges.shape[0]):
            node_a = self.social_edges[i][0]
            node_b = self.social_edges[i][1]

            # Write the activation probability only if the nodes are in range
            if node_a < max_node and node_b < max_node:
                matrix[node_a, node_b] = self.features[i]

        return matrix

    def get_edge_count(self):
        return self.social_edges.shape[0]
