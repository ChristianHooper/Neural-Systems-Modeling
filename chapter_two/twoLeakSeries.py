import numpy as np
import matplotlib.pyplot as plt

'''
Script implements a model having two units in series, each recurrent, excitatory self-connections,
allowing units to stream positive feedback on themselves.
'''

def run():

    time = 100 # Discrete time; differential equation used for continuous time

    y = np.zeros([2, time]) # Row 0 first neuron, second row second downstream neuron
    w = np.array([[0.95, 0.0],[0.5, 0.6]]) # Recurrent leaky weight connections to soma

    x = np.zeros(time); x[10]=1 # Initial dendritic input, set input
    v = np.array([1, 0]) # Initial axonal weighted output connections

    for t in range(1, time): # Length is number of neurons
        y[:,t] = w @ y[:,t-1] + v * x[t] # $y(t)=W\cdot{y(t-1)}+V*x(t)$ LaTeX


    plt.plot(np.arange(time), y[0]) # First neuron
    plt.plot(np.arange(time), y[1]) # Second neuron
    plt.show()

    # Eigen Values & Vectors
    eigen_values, eigen_vectors = np.linalg.eig(w)
    print("Values:\n", eigen_values)
    print("\nVectors:\n", eigen_vectors)


if __name__ == "__main__":
    run()
