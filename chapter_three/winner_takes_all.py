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
# np.exp(-0.5 * ((x/sd)**2))
gaussian = lambda sd, x: np.exp(-(x**2/(2 * sd**2)))
ranges = 51

#light = np.ones(ranges); light[int(ranges*0.33):int(ranges*0.66)] = 3

length = [int(-ranges/2),math.ceil(ranges/2)]

sd = [15, 3] # Standard deviation
ap = [0.3, 1.0] # Amplitude

g_one = np.array([gaussian(sd[0], n,) for n in range(*length)]) * ap[0]
g_two = np.array([gaussian(sd[1], n,) for n in range(*length)]) * ap[1]

dog = (g_two - g_one)
dog_3d = laminate(ranges, dog)
#wave = dog_3d @ light


# Plotting
fig = plt.figure(figsize=(8,3))

ax_2d = fig.add_subplot(1,2,1, title='Light & Response')
ax_2d.plot(np.arange(ranges), dog)


x_axis = np.linspace(1, ranges, ranges)
y_axis = np.linspace(1, ranges, ranges)
x_grid, y_grid = np.meshgrid(x_axis, y_axis)

ax_3d = fig.add_subplot(1,2,2, projection='3d')
ax_3d.plot_surface(x_grid, y_grid, dog_3d, cmap='twilight')


plt.show()