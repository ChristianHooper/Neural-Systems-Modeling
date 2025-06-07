import numpy as np
import matplotlib.pyplot as plt
import math

# Gaussian function: $G(x,\sigma)=\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{x^2}{2\sigma^2}}$
gaussian = lambda s,x: np.exp(-(x**2/(2 * s**2)))
ranges = 11
test = np.zeros(ranges); test[5] = 1
#light = np.ones(ranges); light[45:55] = 3
#print(light)
length = [int(-ranges/2),math.ceil(ranges/2)]
weight_matrix = np.eye(len(range(*length))) # Identity matrix
sd = [0.75, 1.5] # Standard deviation
ap = [0.1, 0.3] # Amplitude

g_one = np.array([int(gaussian(sd[0], n)* 10) for n in range(*length)])
g_two = np.array([int(gaussian(sd[1], n)* 10) for n in range(*length)]) * 0.5

dog = (g_one - g_two)

print(test)
for n in range(len(test)): print(test[5-n:10-n], test[10-n:5-n])



#recurrent_weight = np.array()


# TODO: Laminating function, need to work more efficiency

# Plotting
fig = plt.figure(figsize=(8,3))

ax_2d = fig.add_subplot(1,2,1, title='Light & Response')
ax_2d.plot(np.arange(ranges), dog)

x_axis = np.linspace(1, ranges, ranges)
y_axis = np.linspace(1, ranges, ranges)
x_grid, y_grid = np.meshgrid(x_axis, y_axis)

ax_3d = fig.add_subplot(1,2,2, projection='3d')
#ax_3d.plot_surface(x_grid, y_grid, dog_3d, cmap='twilight')


plt.show()