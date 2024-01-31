import torch
import torch.nn as nn

from typing import List


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out:int, hidden_layers: List[int]) -> None:
        super().__init__()

        h_in = n_in
        layers = list()
        for h_layer in hidden_layers:
            layers.append(nn.Linear(h_in, h_layer))
            layers.append(nn.ReLU())
            h_in = h_layer

        layers.append(nn.Linear(h_in, n_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ONTSMLP(MLP):
    def __init__(self, J: int, T: int, n_features: int, n_hidden: int, hidden_size: int) -> None:
        super().__init__(
            n_in=J * n_features,
            n_out=J * T * 2,
            hidden_layers=[hidden_size] * n_hidden,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        z = super().forward(x)

        if self.training:
            return z
        else:
            return torch.sigmoid(z)
