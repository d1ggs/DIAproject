import matplotlib.pyplot as plt
import numpy as np


def plot_approx_error_point2(result_simulations, plot_name):
    opt = result_simulations.max()
    x = np.arange(1, len(result_simulations)+1)
    y = opt - result_simulations
    print(x,result_simulations)
    # plotting the points  
    plt.plot(x, y) 
    
    # naming the x axis 
    plt.xlabel('MonteCarlo Simulations') 
    # naming the y axis 
    plt.ylabel('Error') 
    
    # giving a title to my graph 
    plt.title('Approximation Error') 
    
    # function to show the plot 
    plt.show() 

    plt.savefig('./social_influence/plots/'+plot_name+'.png')