import numpy as np
import matplotlib.pyplot as plt

'''
The scripts implements the parallel-pathway and positive-feedback models of velocity storage,
and the negative feedback model of velocity leakage.
'''

def run():

    time = 100
    pre_soma = np.array([0.9**x for x in range(time)]) # Input soma state (x), gradiated input fall-off
    pre_axon = np.array([1.0, 0.18]) # Input axon, pre-synaptic signal (v)
    post_axon = np.array([[0, 0.2],[0, 0.95]]) # Output axon, post-synaptic (w)
    soma = np.zeros((2, time)) # Main soma state (y)

    # $y(t)=w\cdot{y(t-1)}+vx(t-1)$ (LaTeX)
    for t in range(1, time): soma[:,t] = post_axon @ soma[:,t-1] + pre_axon[:] * pre_soma[t-1]

    # Plotting
    plt.plot(np.arange(time), pre_soma, label='dot') # Gradiated input (Vestibular afferent neuron)
    plt.plot(np.arange(time), soma[0,:]) # Downstream neuron (Motoneuron)
    plt.plot(np.arange(time), soma[1,:]) # Upstream neuron (vestibular nucleus neuron)
    plt.show()

if __name__ == "__main__":
    run()