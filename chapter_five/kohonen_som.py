import numpy as np
import matplotlib.pyplot as plt


def som(
x: np.ndarray,
v: np.ndarray,
y: np.ndarray,
a: float,
d: float,
cycles: int
) -> np.ndarray:

    for c in range(cycles):
        p_i = np.random.randint(0, len(x))

        y[p_i] = v @ x[p_i] # Define dot-product: $y=Vx$

        print(f'{y[p_i]} = {v} @ {x[p_i]}')

        y_m = np.argmax(y[p_i]) # Finds max neighbourhood: $y_m=max_i(y_i)$
        print(f'Product: {y[p_i]}\nMax Index: {y_m}')

        # Modify & normalize weights: $V_{rowh}(c+1)=\large\frac{V_{rowh}(c)+a x.T(c)}{||V_{rowh}(c+1)=V_{rowh}(c)+a x.T(c)||}$
        v[y_m] = (v[y_m] + a * x[y_m]) / (np.sum(np.power((v[y_m] + a * x[y_m]), 2)) ** 0.5)

        a *= d # Decrement learning rate
    return y


if __name__ == "__main__":

    iteration = 1
    input_neurons = 3
    output_neurons = 3
    learning_rate = 1.0
    learning_decrement = 1.0

    #inputs = np.zeros((iteration, input_neurons))
    #outputs = np.zeros((iteration, output_neurons))
    inputs = np.array([[1,0,0], [0,1,0], [0,0,1]])
    outputs = np.array([[0,0,0], [0,0,0], [0,0,0]], dtype=float)
    weights = np.random.rand(input_neurons, output_neurons)

    training = som(inputs, weights, outputs, learning_rate, learning_decrement, iteration)
    print(training)