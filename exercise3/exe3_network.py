import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, trial, input_dim: int = 1024, output_dim: int = 1):
        super(Model, self).__init__()

        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers = []


        in_features = input_dim
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 4, 512)
            layers.append(nn.Linear(in_features, out_features))
            uf = trial.suggest_categorical("unit_func_{}".format(i), ["ReLU", "LeakyReLU", "SELU"])
            layers.append(getattr(nn, uf)())
            p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
            layers.append(nn.Dropout(p))

            in_features = out_features
        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(
            *layers
        )

        # Play around with some layers https://pytorch.org/docs/stable/nn.html

        self.sigmoid = nn.Sigmoid()

        # TODO add your model config here which are essential for your initialisation
        #  -> the args in __init__()
        self.config = {'trial': trial, 'input_dim': input_dim, 'output_dim': output_dim}

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        return self.sigmoid(self.network(x))

    def get_config(self):
        return self.config.copy()
