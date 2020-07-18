import numpy as np


def seeds_to_binary(seeds, max_nodes):
    for i in seeds:
        tmp = np.zeros(max_nodes)
        tmp[seeds[i]] = 1
        seeds[i] = tmp
    return seeds
