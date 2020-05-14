import matplotlib.pyplot as plt
from typing import List,Dict, Any
from pathlib import Path

def plot_approx_error(result_simulations : List[Dict[str, Any]], infl_max_mc : float, plot_name : str):
    """
    This function plots the approximation error as the parameters of the algorithms vary for every specific network. 
    The parameter is the number of monte_carlo_simulations

    Arguments:

    results_simulations: list where each element is a dict contatining influence and number of mc simulations

    infl_max_mc: influence when the number of mc simulations is max. We compute the error w.r.t this number

    plot_name : name of the plot
    """
    sim_x = []
    infl_y = []
    for r in result_simulations:
        sim_x.append(r["sim"])
        infl_y.append(r["influence"])
    
    infl_y = [y-infl_max_mc for y in infl_y]
    
    # plotting the points  
    plt.plot(sim_x, infl_y) 
    
    # naming the x axis 
    plt.xlabel('MonteCarlo Simulations') 
    # naming the y axis 
    plt.ylabel('Error') 
    
    # giving a title to my graph 
    plt.title('Approximation Error') 
    
    # function to show the plot 
    plt.show() 

    plt.savefig(Path('./social_influence/plots/'+plot_name+'.png'))