import numpy as np
import matplotlib.pyplot as plt



def run():
    
    # Base neural parameters
    time = 101
    neuron_sum = 10

    # Weights set randomly by gaussian distribution
    v = np.random.normal(loc=0, scale=0.5, size=(neuron_sum, neuron_sum)) # Feed-forward weights, every pre-synaptic axon connects to every dendrite
    w = np.random.normal(loc=0, scale=0.5, size=(neuron_sum, neuron_sum)); w *= 0.5 # Feed-back weights, post-synaptic back to pre-synaptic, reduced strength 

    # Neural responses
    x = np.zeros((time, 1, neuron_sum)); x[9, :] = 1 # Creates column for pre-synaptic dendritic input, not transposed
    y = np.zeros((time, 1, neuron_sum)) # Creates column for neural post-synaptic output, 
    
    
    for t in range(0, time): # f(t)=w\cdot{y(t-1)+v\cdot{t-1}}
        y[t] = (w @ y[t-1].T + v @ x[t-1].T).T # Transposes product to fit back in neuron row array
    
    # Plot set-up
    fig = plt.figure(figsize=(8, 8))
    axw = fig.add_subplot(221, projection='3d')
    axv = fig.add_subplot(222, projection='3d')
    x_axis = np.linspace(1, neuron_sum, neuron_sum)
    y_axis = np.linspace(1, neuron_sum, neuron_sum)
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)

    # Post-synaptic axon weight plot
    axw.set_zlim(-1,1)
    axw.plot_surface(x_axis, y_axis, w, cmap='viridis')

    # Pre-synaptic dendrite weight plot
    axv.set_zlim(-1,1)
    axv.plot_surface(x_axis, y_axis, v, cmap='viridis')

        

    fig, axs = plt.subplots(2)
    for i in range(neuron_sum): axs[1].plot(y[:, :, i])
    plt.show()
    



if __name__ == "__main__":
    run()
