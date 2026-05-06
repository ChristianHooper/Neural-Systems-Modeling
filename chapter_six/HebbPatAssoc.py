import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class basic_neuron():
    def __init__(self, inputs):
        self.x = inputs
        self.x_n = inputs.shape[0]
        print(self.x_n)
        self.v = F.normalize(torch.rand((self.x_n, self.x_n), dtype=torch.float32), dim=1)
        self.y = torch.ones(self.x_n)

    def forward(self): self.y = self.v @ self.x

if __name__ == "__main__":

    # Each row represents the dendritic weights carried to all other output neurons

    #hebb = basic_neuron(inputs=inputs)
    #print(hebb.y)

    x = torch.eye(4, dtype=torch.float32) # Inputs
    x[:-1, 0] = 1
    print(x)
    x_in_pnum, x_in_pp = x.shape

    y = torch.unsqueeze(torch.tensor([1, 1, 0, 0], dtype=torch.float32), dim=1)
    print(y.shape)
    y_out_pnum, y_out_pp = y.shape

    V = torch.zeros(x_in_pp, y_out_pp)

    # Sets weights for network based upon hebbian rule; both neurons much be active at the same time
    for i in range(y_out_pnum):
        V[i] = torch.sum(x[i] * y[i])

    print('Weights:\n', V)
    print('Outputs:\n', x @ V)

