import numpy as np
from copy import copy
import os
from const import ROOT_PROJECT_PATH, MATRIX_PATH, FEATURE_MAX
from helper import Helper
from social_setup import SocialNetwork

#this function simulates an episode starting given seeds
#it returns the nodes activated at each time step
def simulate_episode(init_prob_matrix, seeds, n_steps_max):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    assert(seeds.shape[0]==n_nodes)
    history = np.array([seeds])

    active_nodes = seeds
    newly_active_nodes = active_nodes    #node active in the current timestep
    t = 0

    # Loop until either the time is exhausted or there is no new active node
    while(t<n_steps_max and np.sum(newly_active_nodes)>0):
        p = (prob_matrix.T*active_nodes).T  # This is the probability matrix but only with active nodes

        # Find edges exceeding an activation probability threshold and activate them
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])
        # Remove from the probability matrix the probability values related to the previously activated nodes
        prob_matrix = prob_matrix * ((p != 0) == activated_edges)

        # Activate those nodes which have at least and active edge and were not already active,
        # then add them to the currently active nodes
        newly_active_nodes = (np.sum(activated_edges, axis=0) > 0) * (1 - active_nodes)
        active_nodes = np.array(active_nodes + newly_active_nodes)

        history = np.concatenate((history, [newly_active_nodes]), axis=0)
        t += 1
    return history


#dataset: each element is an episode(nodes activated given a fixed seed set)
#it return the activation probabilities of each node
def mc_sampling(dataset, n_nodes):
    z = np.zeros(n_nodes)
    occurr_v_active = np.zeros(n_nodes) #occurencies of each node in all episode
    n_episodes = len(dataset)
    for episode in dataset:
        n_steps = episode.shape[0]
        #activated_nodes = np.argwhere(episode == 1)  ##fix
        #activated_nodes = activated_nodes[3,:]      ##fix
        for i in range(0,n_nodes): #occurency of each node at each episode
            if (len(np.argwhere(episode[:,i]==1))>0):
                z[i]+=1
  
    estimated_prob = z/n_episodes
    return estimated_prob

if __name__ == "__main__": 
    
    #Simulate Social Network
    parameters = np.array([[0.1, 0.3, 0.2, 0.2,0.2],[0.4, 0.1, 0.2, 0.2,0.1],[0.5, 0.1, 0.1, 0.1,0.2]]) #parameters for each social

    # matrix_path = os.path.join(ROOT_PROJECT_PATH,MATRIX_PATH)
    # if os.path.isfile(matrix_path):
    #     prob_matrix = np.load(matrix_path)
    # else:
    
    helper = Helper()
    dataset = helper.read_dataset()
    social = SocialNetwork(dataset,parameters[0], FEATURE_MAX)
    prob_matrix = social.get_matrix()
    n_nodes = prob_matrix.shape[0]
    

    #n_nodes = 50
    #prob_matrix = np.random.uniform(0.0,0.01,(n_nodes,n_nodes))
    
    n_episodes = 10
    seeds = np.random.binomial(1, 0.1, size=(n_nodes))
    

    dataset=[]
    for e in range(0, n_episodes):
        dataset.append(simulate_episode(init_prob_matrix=prob_matrix, seeds=seeds,n_steps_max=10))
    print("Episodes simulated")
    #print(dataset)
    print(dataset[0].shape)

    nodes_probabilities = mc_sampling(dataset,n_nodes)
    print(', '.join([str(d) for d in nodes_probabilities]))


    #node_index = 4 #node target?
    #estimated_prob = estimate_probabilities(dataset=dataset,node_index=node_index,n_nodes=n_nodes)
    #print("True P Matrix ", prob_matrix[:,node_index])
    #print("Estimated P Matrix ", estimated_prob)
    

