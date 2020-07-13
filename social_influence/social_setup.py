import math

import numpy as np
import pandas as pd
import os
import math



class SocialNetwork:
    def __init__(self, dataset, parameters, feature_max, max_nodes=-1):
        """
        Given a social network dataset (create by Helper class) this class computes the edges activation function from the edge feature

        Features in a Social Network could be number of Tag, Share, Like, Message, Comment...

        :param dataset: tuple of edges, features returned by helper.read_dataset
        :param parameters: each social network has a different set of weights for each feature
        :param feature_max : max possible value that a feature can have, used to normalized the edge activation probabilities
        :param max_nodes : max number of nodes for a social network
        """

        self.social_edges, self.features = dataset
        self.parameters = parameters
        self.feature_max = feature_max
        self.max_nodes = max_nodes

        assert(self.parameters.shape == self.features[0].shape)

        # Use math.isclose because np.sum may don't return exactly 1
        assert (math.isclose(np.sum(self.parameters), 1.0, rel_tol=1e-5))

        self.matrix = self.probability_matrix()

    def compute_activation_prob(self, features) -> float:
        """
        This function takes the features of a single edge and computes the activation probability of that edge

        :param features: array of feature of a single edge
        :return prob: edge activation probability
        """
        out = np.dot(self.parameters, features)  # dot product
        prob = out / self.feature_max  # divide by the maximum value of a feature
        return prob

    def probability_matrix(self) -> np.ndarray:
        """
        Compute the edge activation matrix
        """
        max_node = self.social_edges.max()

        #restrict the matrix to self.max_nodes, if specified
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

    def get_matrix(self) -> np.ndarray:
        return self.matrix

    def get_n_nodes(self) -> int:
        return self.matrix.shape[0]

    def get_edge_features_matrix(self) -> np.ndarray:
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

    def get_edge_count(self) -> int:
        return self.social_edges.shape[0]
