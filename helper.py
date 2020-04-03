import pandas as pd
import numpy as np
import os


ROOT_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
FEATURE_MAX = 100

class Helper:

    def __init__(self):
        '''
        Data are loaded from Stanford's SNAP Facebook dataset, where the edges are undirected. This is not a problem,
        since in FB it's reasonable to have undirected graphs, while for IG and Twitter we can assume this graph to be
        directed from first node to tha last.

        '''

        self.social_network_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/facebook_combined.txt"), sep=" ", header=None)

    def write_full_dataset(self):

        social_network = self.social_network_data.to_numpy()
        features = np.random.randint(FEATURE_MAX ,size=(social_network.shape[0], 5))
        data = np.concatenate((social_network, features), axis=1)
        df = pd.DataFrame(data)
        df = df.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"), header= False, index=False)

    def read_dataset(self):
        dataset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/dataset.csv"))
        dataset = dataset.to_numpy()
        social_nodes = dataset[:, [0, 1]]
        features = dataset[:, [2,3,4,5]]
        return  social_nodes, social_nodes, social_nodes, features
if __name__ == "__main__":
    helper = Helper()
    helper.write_full_dataset()