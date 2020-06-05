import numpy as np
from sklearn.preprocessing import normalize


def create_dataset(n_nodes, n_features):
    mask_edges = np.random.binomial(1, 0.05, size=(n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                mask_edges[i, j] = 0
    random_matrix = np.random.rand(n_nodes, n_nodes)
    prob_matrix = random_matrix * mask_edges
    n_edges = np.count_nonzero(prob_matrix)
    # questa matrice è (n_nodes,n_nodes), in ogni entry è salvato il feature vector corrispondente,
    # mi serve per accedere più facilmente ai feature vector
    feature_matrix_edges = np.zeros((n_nodes, n_nodes, n_features))
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            if prob_matrix[i, j] != 0:
                features_vector = normalize(np.atleast_2d(np.random.rand(n_features)))
                feature_matrix_edges[i, j, :] = features_vector

    return prob_matrix, feature_matrix_edges, n_edges


def create_dataset2(n_nodes, n_features, parameter_vector):
    """

    @param n_nodes: numero nodi
    @param n_features: numero features
    @param parameter_vector: theta
    @return probability_matrix
    @return features_matrix: (n_nodes*n_nodes*n_features) matrix of features, each edge is associated with index (i,j),parameter vector for edge (i,j) is features_marix[i,j,:]
    """
    mask_edges = np.random.binomial(1, 0.2, size=(n_nodes, n_nodes))
    # le probabilità della diagonale devono essere zero.
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                mask_edges[i, j] = 0
    n_edges = np.count_nonzero(mask_edges)
    features_matrix = np.zeros((n_nodes, n_nodes, n_features))
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            if mask_edges[i, j] != 0:
                # forse è meglio normalizzare features_vector?
                features_vector = np.atleast_2d(np.random.rand(n_features))
                features_matrix[i, j, :] = features_vector
    probability_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(0, n_nodes):
        for j in range(0, n_nodes):
            probability_matrix[i, j] = np.dot(np.atleast_2d(features_matrix[i, j, :]), parameter_vector)
    return probability_matrix, features_matrix, n_edges
