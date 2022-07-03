from argparse import Namespace
import math
from turtle import forward
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb


import torch.nn.functional as F

from ..conv2d.conv2dmodel import ConvBlock


class AxialLayers(nn.Module):
    def __init__(
        self,
        embedding_dim=8,
        num_dimentions=3,
        num_layers=4,
        embedding_idx=2,
        num_heads=8,
        dropout=0.00,
        residual=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_idx = embedding_idx
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.attentions = nn.Sequential()
        for i in range(self.num_layers):
            # we need setattr so that lightning can find the module
            self.__setattr__(
                "layer_{}".format(i),
                AxialAttention(
                    self.embedding_dim,
                    num_dimentions,
                    self.num_heads,
                    4,
                    embedding_idx,
                    True,
                ),
            )
            self.attentions.add_module(
                "layer_{}".format(i), self.__getattr__("layer_{}".format(i))
            )

            if i != self.num_layers - 1:
                self.do = nn.Dropout(self.dropout)
                self.attentions.add_module(f"do_{i}", self.do)

    def forward(self, x):
        z = self.attentions(x)
        if self.residual:
            out = z + x
        else:
            out = z
        # out = F.relu(out)
        # out = self.do(out)
        return out


class AxialFrameDiscriminator(nn.Module):
    def __init__(self, params: Namespace, embedding_dim=8):
        super().__init__()
        self.params = params

        self.embedding_dim = embedding_dim

        self.embedding = AxialPositionalEmbedding(
            self.embedding_dim, (params.imsize, params.imsize), 1
        )

        self.embedding_encoder = nn.Sequential(
            nn.Conv2d(
                1, self.embedding_dim, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.ReLU(),
        )

        self.attentions = AxialLayers(embedding_dim, 2, 4, 1, 4, 0.1)

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(params.imsize**2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding_encoder(x)
        x = self.embedding(x)
        # ipdb.set_trace()
        x = self.attentions(x)
        x = t.max(x, dim=1)[0]
        x = self.classifier(x)
        return x


class AxialDiscriminator(nn.Module):
    def __init__(self, params: Namespace, embedding_dim=8):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim
        self.embedding_encoder = nn.Sequential(
            nn.Conv2d(
                1, self.embedding_dim, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.ReLU(),
        )
        self.positional_encoder = AxialPositionalEmbedding(
            embedding_dim,
            (params.in_seq_len + params.out_seq_len, params.imsize, params.imsize),
            2,
        )

        self.attentions = AxialLayers(embedding_dim, 3, 4, 2, 4, 0.1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim * self.params.imsize**2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)
        x = self.embedding_encoder(x)
        x = x.view(b, s, self.embedding_dim, h, w)
        # ipdb.set_trace()
        x = self.positional_encoder(x)
        x = self.attentions(x)
        x = t.max(x, dim=1)[0]

        x = self.classifier(x)
        return x


class AxialGenerator(nn.Module):
    def __init__(self, params, embedding_dim=32):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        self.embedding_encoder = nn.Sequential(
            nn.Conv2d(
                1, self.embedding_dim, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.ReLU(),
        )
        self.positional_encoder = AxialPositionalEmbedding(
            embedding_dim, (params.in_seq_len, params.imsize, params.imsize), 2
        )
        self.embedding_decoder = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        self.attentions = AxialLayers(embedding_dim, 3, 6, 2, 16, 0.1)

        # self.noise_layer = GaussianNoise(0.001)

    def forward(self, x: t.Tensor):

        return self.net(x)

    def net(self, x):

        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)
        x = self.embedding_encoder(x)
        x = x.view(b, s, self.embedding_dim, h, w)
        x = self.positional_encoder(x)
        x = self.attentions(x)

        x = x.view(b * s, self.embedding_dim, h, w)
        x = self.embedding_decoder(x)
        x = x.view(b, s, c, h, w)

        # x = x.view(
        #     x.shape[0],
        #     self.params.in_seq_len,
        #     self.params.n_channels,
        #     self.params.imsize,
        #     self.params.imsize,
        # )

        return x
