import numpy as np
import matplotlib.pyplot as plt


def run():

    resolution = 0.1 # Discrete time steps lengths
    ranges = [-25, 25]
    x = np.arange(*ranges, resolution)
    light = np.ones(len(x)); light[15:35]=3
    print(light)

    sigma_one = 1.5 # SD
    sigma_two = 0.75

    # Gaussian: $f(x)=\exp(-\frac{x^2}{2\sigma^2})$
    gaussian_one = np.array( # Large positive spike
        [np.exp(-(point**2 / sigma_one**2)) for point in x]) * 0.5 # Decrease amplitude
    gaussian_two = np.array( # Large dampened spike
        [np.exp(-(point**2 / sigma_two**2)) for point in x])

    dog = gaussian_two - gaussian_one
    print(dog)

    dog_3d = np.array([dog for row in range(len(dog))]) # Creates third dimension


    #output = light * dog
    #gabor = np.array([(1/(2*np.pi*sigma**2)) * np.exp(-(x**2)/(2*sigma**2)) for x in x])

    #print(x)
    # Plotting
    fig = plt.figure(figsize=(8,3))
    ax_2d = fig.add_subplot(1, 2, 1, title="")
    ax_2d.plot(x, dog, label='DOG')

    #ax_2d_2 = fig.add_subplot(1,2,3)
    #ax_2d_2.plot(x,light)

    # 3-D Plotting
    x_grid, y_grid = np.meshgrid(x, x)
    ax_3d = fig.add_subplot(1,2,2, projection='3d', title='')
    ax_3d.plot_surface(x_grid, y_grid, dog_3d, cmap='winter')

    plt.show()

if __name__ == '__main__':
    run()