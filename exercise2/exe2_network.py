import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1):
        super(Model, self).__init__()

        # TODO add your network specification here

    def forward(self, x: torch.Tensor):
        return None

    def predict(self, x: torch.Tensor):
        return None
