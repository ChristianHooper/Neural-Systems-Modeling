import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    run()
