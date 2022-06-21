import torch as t
import torch.nn as nn


class LinearAutoEncoder(t.nn.Module):
    def __init__(self, params):
        super(LinearAutoEncoder, self).__init__()
        self.params = params

        self.encoder = t.nn.Sequential(
            t.nn.Linear(40 * 40, params.embedding_dim), nn.ReLU()
        )

        self.decoder = t.nn.Sequential(
            t.nn.Linear(params.embedding_dim, 40 * 40), nn.ReLU()
        )

    def forward(self, x):
        (b, c, h, w) = x.shape
        x = x.view(b, c, h * w)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
