import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def SOM(x,
    num_units,
    normalize_data = False,
    a=1,
    d=1,
    c=10
    ):

    # Dimensions of incoming patters
    num_pattern, input_dim = x.shape

    # Stores memory of past changes to be returned
    mem_array = np.empty(c, dtype=object)

    # Normalized incoming data if needed
    if normalize_data: x = F.normalize(x)

    # Creates dendritic weight structure
    V = torch.abs(torch.rand(num_pattern, input_dim, dtype=torch.float32)) # Weight structure
    V = F.normalize(V, dim=1) # Normalizes data

    for U in range(c):

        rand_idx = torch.randperm(num_pattern)
        #print(rand_idx)

        for i in rand_idx:
            y = V @ x[i] # Neural product

            print(f'{V}\n')
            print(f'{x[i]}\n')
            print(f'{y}\n')
            # ${y_m=max_i(y_i)}$
            y_m = torch.argmax(y, dim=0) # Index of maximal response

            # Weight change with L-2 normalization
            row_h_n = V[y_m] + (a * x[i])
            row_h_d = torch.sqrt(torch.sum(torch.abs(V[y_m] + (a * x[i]))**2))
            V[y_m] = row_h_n / row_h_d

            a *= d # Leaning rate decay
            #print(V)
        print("\n")
        #mem_array[U] =  V
        plot_learning(x, V) # Plots base pattern and som representations

    return mem_array

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
    torch.manual_seed(1)

    learning_rate = 1
    learning_decay = 1
    training_iteration = 10

    pattern_base =  np.triu(np.tril(np.ones((8, 10), dtype=np.float32), k=4), k=0)
    pattern_data = torch.from_numpy(pattern_base)

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
    '''
    pattern_data = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    pattern_data = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=torch.float32)
    '''

    num_pattern = pattern_data.shape[0]
    num_units = pattern_data.shape[1]

    # Input data & trains network
    som_history = SOM(
        x=pattern_data,
        num_units = num_units, # (Columns) ()
        normalize_data = True,
        a = learning_rate,
        d = learning_decay,
        c = training_iteration
    )

    print(som_history)

