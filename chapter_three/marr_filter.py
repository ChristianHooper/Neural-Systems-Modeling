import numpy as np
import matplotlib.pyplot as plt


def run():
    sigma = 0.25
    x = np.arange(-3,3,0.1)


    gabor = np.array([(1/(2*np.pi*sigma**2)) * np.exp(-(x**2)/(2*sigma**2)) for x in x])

    plt.plot(x, gabor)
    plt.show()

if __name__ == '__main__':
    run()