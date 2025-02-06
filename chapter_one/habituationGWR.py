import numpy as np
import matplotlib.pyplot as plt

'''
Operation & habituation of the gill-withdraw reflex of Aplysia.
'''

def run():

    time = 30
    x = np.zeros(time); x[2::5] = 1 # Pre-synaptic input overtime
    y = np.zeros(time) # Motoneuron output 
    v = 4 # Axonal weight
    d = 0.7 # Axonal weight damping due to repeated stimulation

    for n in range(time): # Calculates neuron response to incoming sensory afferents 
        y[n] = v * x[n] # $f(x)=vx(t)$ LaTeX
        if x[n] == 1: v = v * d # Decreases post-synaptic response from continued stimulation

    # Plotting
    fig, axs = plt.subplots(2)
    axs[0].set_title("Pre-synaptic input")
    axs[0].plot(x)

    axs[1].set_title("Post-synaptic input")
    axs[1].plot(y)
    
    plt.tight_layout(pad=1)
    plt.show()
    

if __name__ == "__main__":
    run()
