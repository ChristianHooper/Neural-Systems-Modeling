import numpy as np
import matplotlib.pyplot as plt

'''
Simple simulation of the pulse or step response of a single neuron with positive feedback.
'''

def run():

    time = 101

    x = np.zeros((time)); x[4]=1 # Pre-synaptic axonal signal
    v = 1 # Pre-synaptic axonal signal

    y = np.zeros((time)) # State of the neuron
    w = 0.95 # Post-synaptic axonal weight


    for n in range(1, time): # Positive feedback of from neuronal response 
        y[n] = x[n-1] * v + y[n-1] * w # $f(t)=g(t-1)v+y(t-1)w$


    # Plotting
    fig, axs = plt.subplots(2)
    axs[0].plot(x, color='r') # Neural input plot overtime
    axs[0].set_title("Pre-synaptic Axonal Input")

    axs[1].set_ylim(0,1)
    axs[1].plot(y) # Neural state plot over time
    axs[1].set_title("Single Neuron Positive Feed-back")
    
    plt.tight_layout(pad=1)
    plt.show()



if __name__ == "__main__":
    run()