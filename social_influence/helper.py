import pandas as pd
import numpy as np
import os
from social_influence.const import FEATURE_MAX, ROOT_PROJECT_PATH
from tqdm import tqdm


class Helper:

    def __init__(self, dataset: str = None):
        '''
        Data are loaded from Stanford's SNAP Facebook dataset, where the edges are undirected. This is not a problem,
        since in FB it's reasonable to have undirected graphs, while for IG and Twitter we can assume this graph to be
        directed from first node to the last.

        '''
        if dataset != None:
            self.social_network_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset + ".txt"), sep=" ", header=None)

    def write_full_dataset(self, dataset):

        social_network = self.social_network_data.to_numpy()
        features = np.random.randint(FEATURE_MAX ,size=(social_network.shape[0], 5))
        data = np.concatenate((social_network, features), axis=1)
        df = pd.DataFrame(data)
        df = df.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset + ".csv"), header= False, index=False)

    def read_dataset(self, dataset):
        dataset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset + ".csv"))
        dataset = dataset.to_numpy()
        social_nodes = dataset[:, :2]
        features = dataset[:, 2:]
        return social_nodes,  features

if __name__ == "__main__":
    helper = Helper("gplus_combined")
    helper.write_full_dataset("gplus")
