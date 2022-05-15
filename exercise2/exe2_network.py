import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1):
        super(Model, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.SELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x: torch.Tensor):
        return self.network(x)

    def predict(self, x: torch.Tensor):
        return self.sigmoid(self.network(x))
