import numpy as np
import matplotlib.pyplot as plt

'''
Script implements the two-unit model; of the integrator of the oculomotor system.
'''

def run():

    time = 1000
    bg = 10 # Background noise

    x = np.ones((2, time)) * bg; x[0,100]+=1; x[1,100]-=1 # Pre-synaptic inputs
    y = np.zeros((2, time)) # Neuron state

    v = np.array([[1, 0], [0, 1]]) # Pre-synaptic input weight
    w = np.array([[0.5, -0.499], [-0.499, 0.5]]) # Post-synaptic output weight, negative inhibition & positive self recurrent connections

    # Integrator equation
    for t in range(time): y[:,t] = w @ y[:,t-1] + v @ x[:,t-1]

    # Eigen values
    eigen_values, eigen_vectors = np.linalg.eig(w)
    print("Values:\n", eigen_values)
    print("\nVectors:\n", eigen_vectors,'\n')

    plt.plot(np.arange(time), y[0,:],)
    plt.plot(np.arange(time), y[1,:])
    plt.plot(np.arange(time), x[0,:], ':')
    plt.plot(np.arange(time), x[1,:], ':')
    plt.show()

if __name__ == "__main__":
    run()
