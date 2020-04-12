import numpy as np
from copy import copy



class MonteCarloSampling(object):
    """
    Attributes
    --------
    edge_activations : edge activation functions matrix
    """
    def __init__(self, edge_activations):
        super().__init__()
        self.edge_activations = edge_activations
        self.n_nodes = edge_activations.shape[0]

    def simulate_episode(self, seeds, n_steps_max):
        """
        Simulates an episode where at each time step certain nodes activates

        Parameters
        --------

        seeds : initial set of seed nodes

        n_steps_max : number of time steps inside one episode
        """
        prob_matrix = self.edge_activations.copy()
        assert(seeds.shape[0]==self.n_nodes)
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


    def mc_sampling(self, seeds, n_episodes, n_steps_max):
        """
        Implements Monte Carlo Sampling, from the edge probabilities and a given set of seeds, it returns 
        the node activation functions

        Parameters
        --------
        seeds : set of seed nodes

        n_episode : number of episodes

        n_steps_max : max number of steps in a episode
        """
        z = np.zeros(self.n_nodes)
        occurr_v_active = np.zeros(self.n_nodes) #occurencies of each node in all episode

        for i in range(n_episodes):
            episode = self.simulate_episode(seeds=seeds, n_steps_max=n_steps_max)
            n_steps = episode.shape[0]
            
            for i in range(0,self.n_nodes): #occurency of each node at each episode
                if (len(np.argwhere(episode[:,i]==1))>0):
                    z[i]+=1
    
        estimated_prob = z/n_episodes
        return estimated_prob