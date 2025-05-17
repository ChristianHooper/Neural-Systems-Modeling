import numpy as np
import matplotlib.pyplot as plt

# Lateral inhibition edge detection through connectivity matrix

size = 50 # Number of input stimulus'
light_size = 10 # Light is twice the size

x = np.ones(size) # Declares 1-D light vector
light_index = int(size/2) # Position on vector for defining light
x[light_index-light_size : light_index+light_size] = 2 # Defines 1-D light vector

print("Light", x)

v = np.zeros((size, size)) # Lateral inhibition weights

# Sets up neural weights
offsets = [-1, 0, 1]
weights = [-1, 2, -1]
for di in range(size): # Create diagonal weight structure
    for off, w in zip(offsets, weights):
        index = di + off
        if -1 <= index < size:
            v[di, index] = w
        else: v[di, 0] = -1 # Loops last row value

y = v @ x # Lateral inhibition to define edge contrast from light

fig = plt.figure(figsize=(8,3))

ax_2d = fig.add_subplot(1,2,1, title='Light & Response')
ax_2d.plot(np.arange(size), y, label='Neural Response')
ax_2d.plot(np.arange(size), x, ':', label='Light Dimension')
fig.legend(title='Box')

x_axis = np.linspace(1, size, size)
y_axis = np.linspace(1, size, size)
x_grid, y_grid = np.meshgrid(x_axis, y_axis)

ax_3d = fig.add_subplot(1,2,2, projection='3d', title='Weights')
ax_3d.plot_surface(x_grid, y_grid, v, cmap='hot')

plt.show()
