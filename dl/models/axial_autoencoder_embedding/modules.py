import torch as t
import torch.nn as nn
import ipdb

class AutoEncoder(nn.Module):
    def __init__(self, embedding_dim=256) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(4096, embedding_dim, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 4096, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, s, c, h, w = x.size()
        x = x.view(x.size(0), x.size(1), x.size(2), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(b, s, c, h, w)
        return x
