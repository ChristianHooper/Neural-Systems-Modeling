import numpy as np
import matplotlib.pyplot as plt


field_size = 32 # Area size

light_size = [int(field_size/4), int(field_size*0.75)] # Size of light on area

light = np.zeros(field_size); # Declares light
light[light_size[0]:light_size[1]] = 1 # Defines light


def identity(order=0): # Create an identify matrix for matrix multiplication
    matrix_space = np.zeros((field_size, field_size))
    pattern = {0:(0,0), 1:(-1,1), 2:(1,-2,1)}

    if order == 0: return matrix_space # Default return

    # How many indices to shift to the left in diagonal render
    shift = len(pattern[order]) - len(pattern[order-1])

    for i, row in enumerate(matrix_space):
        for n in range(len(pattern[order])):
            index = n + i - shift # Shift to correct placement
            if index >= len(row): index = (index-len(row))*-1 # Error correction bottom matrices
            row[index] = pattern[order][n] # Sets each indices
    return matrix_space

# Creates identities needed for multiplication
first_order = identity(1)
second_order = identity(2)

first_derivative = first_order @ light # First order derivative calculation
second_derivative = second_order @ light # First order derivative calculation

# Plot contents
x_axis = np.arange(0, field_size, 1)
fig, axs = plt.subplots(3, sharex=True)
axs[0].plot(x_axis, light)
axs[1].plot(x_axis, first_derivative)
axs[2].plot(x_axis, second_derivative)

plt.show()
