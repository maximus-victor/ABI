import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1):
        super(Model, self).__init__()

        # TODO add your network specification here

        further_dims = output_dim or torch.randint(50, 300, (input_dim - 1,)).tolist()
        in_dims = [45] + further_dims
        out_dims = further_dims + [1]

        linear_layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, out_dims)]
        activations = [nn.LeakyReLU() for _ in range(input_dim - 1)]

        nn_layers = [None] * (len(linear_layers) + len(activations))
        nn_layers[::2], nn_layers[1::2] = linear_layers, activations

        self.network = nn.Sequential(
            *nn_layers
        )

        # Play around with some layers https://pytorch.org/docs/stable/nn.html

        self.sigmoid = nn.Sigmoid()

        # TODO add your model config here which are essential for your initialisation
        #  -> the args in __init__()
        self.config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        return self.sigmoid(self.network(x))

    def get_config(self):
        return self.config.copy()
