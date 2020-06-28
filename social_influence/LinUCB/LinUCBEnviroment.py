import numpy as np


class LinUCBEnviroment():
    def __init__(self, probability_matrix):
        self.probability_matrix = probability_matrix

    def round(self, pulled_arm):
        p = np.random.random()
        if self.probability_matrix[pulled_arm[0], pulled_arm[1]] > p:
            return 1
        else:
            return 0


    def opt(self):
        return np.max(self.probability_matrix)
