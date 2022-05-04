import math
from turtle import forward
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb
from ...components.components import GaussianNoise


class AxialGenerator(nn.Module):
    def __init__(self, params, embedding_dim=64):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        # embedding encoder
        self.embedding_encoder = nn.Sequential(
            nn.Linear(1, embedding_dim, bias=True),
            # what activation should we use?
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.image_embedding = nn.Sequential(
            nn.Linear(4096, embedding_dim, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # decoding the embedding to the output
        self.embedding_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 4096, bias=True), nn.Sigmoid()
        )

        self.positional_embedding = AxialPositionalEmbedding(
            embedding_dim,
            (params.in_seq_len, params.n_channels),
            -1,
        )

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.10),
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention,
                dim_heads=4,
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention,
                dim_heads=4,
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
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
        # ipdb.set_trace()

        self.noise_layer = GaussianNoise(0.001)

    def forward(self, x: t.Tensor):

        return self.net(x)

    def net(self, x):

        x = self.noise_layer(x)
        # if self.training:
        #     x = self.noise_layer(x)
        # else:
        #     x = self.noise_layer(x, 0.0001)

        x = x.unsqueeze(-1)
        # ipdb.set_trace()
        
        x = x.view(x.shape[0], self.params.in_seq_len, self.params.n_channels, -1)
        
        x = self.image_embedding(x)

        x = self.positional_embedding(x)

        x = self.attentions(x)

        x = self.embedding_decoder(x)

        x = x.view(x.shape[0], self.params.in_seq_len, self.params.n_channels, self.params.imsize, self.params.imsize)

        return x
