import os
import numpy as np
#path
ROOT_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

#dataset
FEATURE_MAX = 100
FEATURE_PARAM = np.asarray(
    ((0.1, 0.3, 0.2, 0.2, 0.2), (0.3, 0.1, 0.2, 0.2, 0.2), (0.5, 0.1, 0.1, 0.1, 0.2)))  # parameters for each social
SOCIAL_NAMES = ["gplus", "facebook", "wikipedia"]
MAX_NODES = 1000