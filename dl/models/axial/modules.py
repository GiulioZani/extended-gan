import math
from turtle import forward
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb
from ...components.components import GaussianNoise


class AxialGenerator(nn.Module):
    def __init__(self, params, embedding_dim=8):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        # embedding encoder
        self.embedding_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim, bias=True),
            # what activation should we use?
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoding the embedding to the output
        self.embedding_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 1, bias=True), nn.Sigmoid()
        )

        self.positional_embedding = AxialPositionalEmbedding(
            embedding_dim,
            (params.in_seq_len, params.n_channels, params.imsize, params.imsize),
            -1,
        )

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=4,  # number of heads for multi-head attention
                # dim_heads=4,
                num_dimensions=4,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.35),
            # AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=4,  # number of heads for multi-head attention,
            #     # dim_heads=4,
            #     num_dimensions=4,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.LeakyReLU(0.2),
            # AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=4,  # number of heads for multi-head attention,
            #     dim_heads=4,
            #     num_dimensions=4,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.LeakyReLU(0.2),
            # AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=8,  # number of heads for multi-head attention,
            #     dim_heads=4,
            #     num_dimensions=4,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.LeakyReLU(0.2),
            # AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=8,  # number of heads for multi-head attention,
            #     dim_heads=4,
            #     num_dimensions=4,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.LeakyReLU(0.2),
        )

        self.noise_layer = GaussianNoise(0.001)

    def forward(self, x: t.Tensor):

        return self.net(x)

    def net(self, x):

        if self.training:
            x = self.noise_layer(x)
        else:
            x = self.noise_layer(x, 0.0001)

        x = x.unsqueeze(-1)

        x = self.embedding_encoder(x)

        x = self.positional_embedding(x)

        x = self.attentions(x)

        x = self.embedding_decoder(x)


        return x.squeeze(-1)
