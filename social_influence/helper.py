import pandas as pd
import numpy as np
import os
from social_influence.const import FEATURE_MAX, ROOT_PROJECT_PATH
from tqdm import tqdm


class Helper:

    def __init__(self, dataset: str = None):
        """
        This class converts the raw data of a social graph into a social network with edge features

        :param dataset: this string indicates the dataset

        """
        if dataset is not None:
            self.social_network_data = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset + ".txt"), sep="\s+", header=None)

    def write_full_dataset(self, dataset: str):

        social_network = self.social_network_data.to_numpy()
        features = np.random.randint(FEATURE_MAX, size=(social_network.shape[0], 5))
        data = np.concatenate((social_network, features), axis=1)
        df = pd.DataFrame(data, columns=["source", "destination", "f1", "f2", "f3", "f4", "f5"])
        df = df.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset + ".csv"), header=True, index=False)

    def read_dataset(self, dataset: str):
        dataset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset + ".csv"))
        dataset = dataset.to_numpy()
        social_nodes = dataset[:, :2].astype(int)
        features = dataset[:, 2:]
        return social_nodes, features

    @staticmethod
    def convert_dataset(dataset_name: str):
        """
        The method converts the selected dataset, reducing the length of the indexes

        :param dataset_name: the string identifying the dataset, without file extension
        """

        # Load the dataset
        dataset = pd.read_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset_name + ".csv"))

        # Build a list of unique indexes in the dataset
        indexes = list(dataset["source"].unique())
        indexes.extend(list(dataset["destination"].unique()))
        indexes = list(set(indexes))

        # Build a dictionary remapping the old indexes to (1, 2, ..., MAX)
        mapping = {k: v for k, v in zip(indexes, range(len(indexes)))}

        # Prepare the new columns for the dataset source and destination nodes
        new_source = []
        new_destination = []
        for s, d in zip(dataset["source"], dataset["destination"]):
            new_source.append(mapping[s])
            new_destination.append(mapping[d])

        # Update and store the dataset
        dataset["source"] = new_source
        dataset["destination"] = new_destination
        dataset.to_csv(os.path.join(ROOT_PROJECT_PATH, "data/" + dataset_name + "_fixed.csv"), header=True, index=False)



if __name__ == "__main__":
    helper = Helper("wikipedia")
    # helper.read_dataset("gplus")
    # helper.convert_dataset("gplus")
    # helper.convert_dataset("facebook")
    # helper.convert_dataset("twitter")
    helper.convert_dataset("wikipedia")

