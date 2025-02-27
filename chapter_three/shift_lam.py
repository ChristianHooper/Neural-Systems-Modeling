import numpy as np
import matplotlib.pyplot as plt

# Lateral inhibition edge detection

size = 7 # Number of input stimulus'

x = np.array([1,1,2,2,2,1,1]) # 1-D light vector

v = np.zeros((size, size)) # Lateral inhibition weights

offsets = [-1, 0, 1]
weights = [-1, 2, -1]
for di in range(size): # Create diagonal wight structure
    for off, w in zip(offsets, weights):
        index = di + off
        if -1 <= index < size:
            v[di, index] = w
        else: v[di, 0] = -1 # Loops last row value

y = v @ x

print(y)


