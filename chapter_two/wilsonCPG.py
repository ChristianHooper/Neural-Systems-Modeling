import numpy as np
import matplotlib.pyplot as plt

'''
Script implements a linear version of Wilson's model of the locust-flight central pattern generator oscillation indicates fatigue.
'''

time = 100
fly = 10 # Sets time to begin flight oscillations

y = np.zeros((4,time)) # Neural states
x = np.zeros((1,time)); x[0,fly]=1 # Pre-synaptic input

w = np.array([ # Post-synaptic weights
    [0.9,    0.2,  0,    0],
    [-0.95,  0.4, -0.5,  0],
    [0,     -0.5,  0.4, -0.95],
    [0,      0,    0.2,  0.9]])

v = np.array([0, 1, 0, 0]) # Pre-synaptic weights

for t in range(time): y[:,t] = w @ y[:,t-1] + v * x[:,t-1] # Neural interaction

plt.plot(np.arange(time), y[0,:], ':', label='Recurrent Leak-Integrator y_4')
plt.plot(np.arange(time), y[1,:], label='Oscillation Motor Inhibitor y_2')
plt.plot(np.arange(time), y[2,:], label='Oscillation Motor Inhibitor y_3')
plt.plot(np.arange(time), y[3,:], ':', label='Recurrent Leak-Integrator y_4')
plt.legend(title='Neurons Function')
plt.show()
