import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1):
        super(Model, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def predict(self, x: torch.Tensor):
        return nn.Sigmoid(self.network(x))
