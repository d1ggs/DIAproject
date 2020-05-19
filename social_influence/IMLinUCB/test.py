from social_influence.IMLinUCB.IMLinUCBEnviroment import *
from social_influence.IMLinUCB.IMLinUCBLearner import *
from social_influence.IMLinUCB.create_dataset import *
import matplotlib.pyplot as plt

budget = 2
n_nodes = 20
n_features = 5
T = 50
n_experiment = 1
parameters = np.array([0.7, 0.01,0.3,0.1,0.2])
prob_matrix, features_edge_matrix, n_edges = create_dataset2(n_nodes, n_features, parameters)
reward_per_experiment = []


for exp in range(n_experiment):
    env = IMLinUCBEnviroment(prob_matrix,budget)
    learner = IMLinUCBLearner(n_features, features_edge_matrix, budget)
    for t in range(T):
        pulled_arm = learner.pull_arm()
        reward, edge_activation_matrix,seen_edges = env.round(pulled_arm)
        learner.update_observations(reward, edge_activation_matrix,seen_edges)

    reward_per_experiment.append(learner.collected_rewards)

#sistemo sintassi
another_env = IMLinUCBEnviroment(prob_matrix,budget)
optimal_reward = another_env.opt()

plt.figure()
plt.xlabel("Time")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(optimal_reward - reward_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()

