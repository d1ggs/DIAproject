import pandas as pd
import numpy as np
import os
from const import FEATURE_MAX, ROOT_PROJECT_PATH



class Helper:

    def __init__(self):
        '''
        Data are loaded from Stanford's SNAP Facebook dataset, where the edges are undirected. This is not a problem,
        since in FB it's reasonable to have undirected graphs, while for IG and Twitter we can assume this graph to be
        directed from first node to the last.

        '''

        self.social_network_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/nodes.txt"), sep=" ", header=None)
        self.social_network = self.social_network_data.to_numpy()
        self.feature_max = FEATURE_MAX

    def write_full_dataset(self):

        interaction_features_facebook = np.random.randint(self.feature_max ,size=(self.social_network.shape[0], 5))
        interaction_features_instagram = np.random.randint(self.feature_max ,size=(self.social_network.shape[0], 5))
        interaction_features_twitter = np.random.randint(self.feature_max ,size=(self.social_network.shape[0], 5))
        df_interaction_features_facebook = pd.DataFrame(interaction_features_facebook)
        df_interaction_features_instagram = pd.DataFrame(interaction_features_instagram)
        df_interaction_features_twitter = pd.DataFrame(interaction_features_twitter)
        df_interaction_features_facebook = df_interaction_features_facebook.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/interaction_features_facebook.csv"), header= False, index=False)
        df_interaction_features_instagram = df_interaction_features_instagram.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/interaction_features_instagram.csv"), header= False, index=False)
        df_interaction_features_twitter = df_interaction_features_twitter.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/interaction_features_twitter.csv"), header= False, index=False)

    def get_social_nodes(self):
        return self.social_network

    def get_feature_max(self):
        return self.feature_max

    def get_interaction_features(self, social):
        print("Reading interaction features of social " + social)
        interaction_features = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/interaction_features_" + social + ".csv"))
        interaction_features = interaction_features.to_numpy()
        return  interaction_features

if __name__ == "__main__":
    helper = Helper()
    helper.write_full_dataset()