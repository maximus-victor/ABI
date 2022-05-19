import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1):
        super(Model, self).__init__()

        # TODO add your network specification here

        # TODO add your model config here which are essential for your initialisation
        #  -> the args in __init__()
        self.config = {'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, x: torch.Tensor):
        return None

    def predict(self, x: torch.Tensor):
        return None

    def get_config(self):
        return self.config.copy()
