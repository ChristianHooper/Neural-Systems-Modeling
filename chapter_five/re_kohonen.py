import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def SOM(x,
    p_o,
    p_i,
    normalize_data = False,
    a=1,
    d=1,
    c=10
    ):

    # Normalized incoming data if needed
    if normalize_data: x = F.normalize(x)

    # Creates dendritic weight structure
    V = torch.randn((p_o, p_i), dtype=torch.float32) # Weight structure

    # TODO: Working will small patterns, don't seem to work on larger pattern well
    # TODO: Break a part the 2-loop structure and relay only on one if possible; figure that math out
    for _ in range(c):
        # $\large{y=Vx}$
        for i in range(p_o):
            y = V @ x[i] # Neural product

            # $\large{y_m=max_i(y_i)}$
            y_m = torch.argmax(y, dim=0) # Index of maximal response

            # Weight change with L-2 normalization
            row_h_n = V[y_m] + (a * x[i])
            row_h_d = torch.sqrt(torch.sum(torch.abs(V[y_m] + (a * x[i]))**2))
            V[y_m] = row_h_n / row_h_d

            a *= d # Leaning rate decay
        plot_learning(x, V) # Plots base pattern and som representations


def plot_learning(x, V):

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(x, cmap="bwr")
    axes[0].set_title("Base Pattern")
    axes[0].axis("off")

    axes[1].imshow(V, cmap="bwr")
    axes[1].set_title("SOM Representation")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(0)

    pattern_data = torch.tensor([
        [5, 0, 0],
        [4, 1, 0],
        [3, 1, 1],
        [0, 5, 0],
        [0, 4, 1],
        [1, 3, 1],
        [0, 0, 5],
        [0, 1, 4],
        [1, 1, 3]
    ], dtype=torch.float32)

    pattern_data = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    learning_rate = 1
    learning_decay = 1
    training_iteration = 100

    # Input data & trains network
    SOM(
        x=pattern_data,
        p_o=len(pattern_data),
        p_i=len(pattern_data[1]),
        normalize_data = False,
        a = learning_rate,
        d = learning_decay,
        c = training_iteration
    )

