import numpy as np
import matplotlib.pyplot as plt


time = 100
fly = 10 # Sets time to begin flight oscillations

y = np.zeros((4,time)) # Neural states
x = np.zeros((1,time)); x[0,fly]=1 # Pre-synaptic input

print(x)

w = np.array([ # Post-synaptic weights
    [0.9,    0.2,  0,    0],
    [-0.95,  0.4, -0.5,  0],
    [0,     -0.5,  0.4, -0.95],
    [0,      0,    0.2,  0.9]])

v = np.array([0, 1, 0, 0]) # Pre-synaptic weights

for t in range(time): y[:,t] = w @ y[:,t-1] + v * x[:,t-t]

#print(y)

plt.plot(np.arange(time), y[0,:])
plt.plot(np.arange(time), y[1,:])
plt.plot(np.arange(time), y[2,:])
plt.plot(np.arange(time), y[3,:])
plt.show()

