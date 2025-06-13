import numpy as np
import matplotlib.pyplot as plt
import math

# Takes a matrix size, and a key to diagonally impresses onto a matrix
def laminate(size=8, key=[-1,1,-1]): # Key must be an old length
    mm = (len(key)-1)/2 # Half size of key for insertion
    key_pattern = np.arange(-mm, mm+1)
    base = np.zeros((size,size))

    for n in range(len(base)):
        for i, offset in enumerate(key_pattern):
            # Impresses patterns key and loops edges
            base[n, int((n + offset)%len(base))] = key[i]
    return np.array(base)


# Gaussian function: $G(x,\sigma)=\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{x^2}{2\sigma^2}}$
gaussian = lambda sd, x: np.exp(-(x**2/(2 * sd**2)))
ranges = 51

#light = np.ones(ranges); light[int(ranges*0.33):int(ranges*0.66)] = 3

length = [int(-ranges/2),math.ceil(ranges/2)]

sd = [15, 3] # Standard deviation
ap = [0.3, 1.0] # Amplitude

g_one = np.array([gaussian(sd[0], n,) for n in range(*length)]) * ap[0]
g_two = np.array([gaussian(sd[1], n,) for n in range(*length)]) * ap[1]
dog = (g_two - g_one)
dog_3d = laminate(ranges, dog) # Weight structure

sin_input = 2 * np.sin(np.pi * np.arange(ranges)/(ranges-1))
sin_input_T = sin_input.reshape(-1,1)

nt = 20 # Time steps
rate = 1.0
cut = 0.0 # Min
sat = 10.0 # Max

W = dog_3d # Weights
x = sin_input # Input
y = np.zeros((ranges, nt)) # Activity of neurons

# Winner-takes-all
for t in range(1, nt):
    y[:, t] = rate * (W @ y[:, t - 1]) + x
    y[:, t] = np.clip(y[:, t], cut, sat) # Threshold


# Plotting
fig = plt.figure(figsize=(10,4))

# 2D dog weight plot
ax_2d = fig.add_subplot(1,3,1, title='Weight')
ax_2d.plot(np.arange(ranges), dog)

# 3D dog weight plot
x_axis = np.linspace(1, ranges, ranges)
y_axis = np.linspace(1, ranges, ranges)
x_grid, y_grid = np.meshgrid(x_axis, y_axis)
ax_3d = fig.add_subplot(1,3,2, projection='3d', title='Weight 3D')
ax_3d.plot_surface(x_grid, y_grid, dog_3d, cmap='twilight')

# WTA plot
ax_3d_2 = fig.add_subplot(1,3,3, projection='3d', title='Winner-Takes-All')
x_t, y_n = np.meshgrid(np.arange(nt), np.arange(ranges))
ax_3d_2.plot_surface(x_t, y_n, y, cmap='twilight')

plt.show()
