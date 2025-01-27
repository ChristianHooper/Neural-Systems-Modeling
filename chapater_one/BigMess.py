import numpy as np
import matplotlib.pyplot as plt

'''
Scripts simulates the pulse or step response of a neural network with ten input & ten output neurons randomly connected.
'''

def run():
    
    # Base neural parameters
    time = 101
    neuron_sum = 10

    # Weights set randomly by gaussian distribution
    v = np.random.normal(loc=0, scale=0.5, size=(neuron_sum, neuron_sum)) # Feed-forward weights, every pre-synaptic axon connects to every dendrite
    w = np.random.normal(loc=0, scale=0.5, size=(neuron_sum, neuron_sum)) # Feed-back weights, post-synaptic back to pre-synaptic, reduced strength 

    pulse = [15] # Times for pre-synaptic input pulse

    # Neural responses
    x = np.zeros((time, neuron_sum)) # Creates column for pre-synaptic dendritic input, not transposed 
    for t in pulse: x[t,:] = 1 
    y = np.zeros((time, neuron_sum)) # Creates column for neural state, used in post-synaptic output/feed-back 


    for t in range(0, time): # f(t)=w\cdot{y(t-1)+v\cdot{t-1}} (LaTeX)
        y[t, :] = w @ y[t-1, :] + v @ x[t-1, :] # Uses matrix multiplication to set discrete state of the neuron
        y[t] = np.clip(y[t], a_min=0, a_max=1000) # Caps feed-back

    #///////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Plot set-up
    fig = plt.figure(figsize=(8, 8))
    axw = fig.add_subplot(221, projection='3d') # Input weight plot
    axv = fig.add_subplot(222, projection='3d') # Output weights plot
    axx = fig.add_subplot(223, projection='3d') # Pre-synaptic plot
    axy = fig.add_subplot(224, projection='3d') # Post-synaptic plot
    
    # For weights plot
    x_axis = np.linspace(1, neuron_sum, neuron_sum)
    y_axis = np.linspace(1, neuron_sum, neuron_sum)

    # For neural responses plot
    x_axis_large = np.linspace(1, neuron_sum, neuron_sum)
    y_axis_large = np.linspace(1, time, time) 

    # Plot axises
    x_axis, y_axis = np.meshgrid(x_axis, y_axis)
    x_axis_large, y_axis_large = np.meshgrid(x_axis_large, y_axis_large)

    print(x_axis_large[10])
    print(x[10])
    # Plotting
    # Post-synaptic axon weight plot
    axw.set_title("Post-Synaptic Axon Weight")
    axw.set_zlim(-1,1)
    axw.plot_surface(x_axis, y_axis, w, cmap='viridis')

    # Pre-synaptic dendrite weight plot
    axv.set_title("Pre-Synaptic Dendrite Weight")
    axv.set_zlim(-1,1)
    axv.plot_surface(x_axis, y_axis, v, cmap='viridis')

    # Pre-Synaptic Inputs
    axx.set_title("Pre-Synaptic Signal Inputs")
    axx.plot_surface(x_axis_large, y_axis_large, x, cmap='viridis')
    axx.set_xlabel("Neuron")
    axx.set_ylabel("Time")
    
    # Neuron State: Post-Synaptic Feed-Back
    axy.set_title("Neuron State: Post-Synaptic Feed-Back")
    axy.plot_surface(x_axis_large, y_axis_large, y, cmap='viridis')
    axy.set_xlabel("Neuron")
    axy.set_ylabel("Time")

    plt.show()
    
    #///////////////////////////////////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    run()
