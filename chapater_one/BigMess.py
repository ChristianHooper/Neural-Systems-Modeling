import numpy as np
import matplotlib.pyplot as plt



def run():
    
    # Base neural parameters
    time = 101
    neuron_sum = 10

    # Weights
    v = np.random.rand(neuron_sum, neuron_sum) # Feed-forward weights, every pre-synaptic axon connects to every dendrite
    w = np.random.rand(neuron_sum, neuron_sum); w *= 0.5 # Feed-back weights, post-synaptic back to pre-synaptic, reduced strength 


    # Neural responses
    x = np.zeros((time, 1, neuron_sum)); x[9, :] = 1 # Creates column for pre-synaptic dendritic input, not transposed
    y = np.zeros((time, 1, neuron_sum)) # Creates column for neural post-synaptic output, 
    
    
    print(v)
    
    for t in range(0, time):
        y[t] = (w @ y[t-1].T + v @ x[t-1].T).T # Transposes product to fit back in neuron row array
        
    

    
    fig, axs = plt.subplots(2)
    axs[1].set_ylim(0, 1000)
    
    for i in range(neuron_sum): axs[1].plot(y[:, :, i])
    plt.show()
    



if __name__ == "__main__":
    run()
