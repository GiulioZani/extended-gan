import math
from turtle import forward
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb

from ..axial_autoencoder_embedding.modules import AutoEncoder
from ...components.components import GaussianNoise


class AxialGenerator(nn.Module):
    def __init__(self, params, embedding_dim=128):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        self.generator = AutoEncoder(embedding_dim=self.embedding_dim)

        # load auto encoder weights
        self.load_state_dict(
            t.load("./dl/models/axial_autoencoder_embedding/checkpoint.ckpt")
        )

        self.embedding_encoder = self.generator.encoder
        self.embedding_decoder = self.generator.decoder

        # disable encoder decoder backprop
        for param in self.embedding_encoder.parameters():
            param.requires_grad = False
        for param in self.embedding_decoder.parameters():
            param.requires_grad = False

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.10),
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention,
                dim_heads=4,
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.Dropout(0.10),
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
            #     num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.LeakyReLU(0.2),
        )

        self.noise_layer = GaussianNoise(0.001)

    def forward(self, x: t.Tensor):

        return self.net(x)

    def net(self, x):

        x = self.noise_layer(x)
        b, s, c, h, w = x.size()
        # ipdb.set_trace()
        x = x.view(x.size(0), x.size(1), x.size(2), -1)
        x = self.embedding_encoder(x)
        x = self.attentions(x)
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


class AxialTemporalDiscriminator(nn.Module):
    def __init__(self, params, embedding_dim=128):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        # embedding encoder
        self.image_embedding = nn.Sequential(
            nn.Linear(4096, embedding_dim, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoding the embedding to the output
        self.embedding_decoder = nn.Sequential(
            nn.Linear(embedding_dim * params.n_channels * 20, 1, bias=True),
            nn.Sigmoid(),
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
            # AxialAttention(
            #     dim=self.embedding_dim,  # embedding dimension
            #     dim_index=-1,  # where is the embedding dimension
            #     heads=8,  # number of heads for multi-head attention,
            #     dim_heads=4,
            #     num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            # ),
            # nn.LeakyReLU(0.2)
        )

        self.noise_layer = GaussianNoise(0.001)

    def forward(self, x: t.Tensor):

        return self.net(x)

    def net(self, x):

        x = self.noise_layer(x)
        if self.training:
            x = self.noise_layer(x)
        else:
            x = self.noise_layer(x, 0.0001)

        x = x.unsqueeze(-1)

        x = x.view(x.shape[0], x.shape[1], self.params.n_channels, -1)
        x = self.image_embedding(x)
        x = self.attentions(x)

        # x = t.max(x, dim=1)[0]
        x = x.view(x.shape[0], -1)

        x = self.embedding_decoder(x)

        return x


class AxialFrameDiscriminator(nn.Module):
    def __init__(self, params, embedding_dim=32):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        # embedding encoder
        self.embedding_encoder = nn.Sequential(
            nn.Linear(4096, embedding_dim, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoding the embedding to the output
        self.embedding_decoder = nn.Sequential(
            nn.Linear(embedding_dim * params.n_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self.attentions = nn.Sequential(
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.10),
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention,
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            # nn.LeakyReLU(0.2),
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention,
                dim_heads=4,
                num_dimensions=1,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            # nn.LeakyReLU(0.2)
        )

        self.noise_layer = GaussianNoise(0.001)

    def forward(self, x: t.Tensor):

        return self.net(x)

    def net(self, x):

        x = self.noise_layer(x)
        if self.training:
            x = self.noise_layer(x)
        else:
            x = self.noise_layer(x, 0.0001)

        # ipdb.set_trace()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.embedding_encoder(x)
        x = self.attentions(x)

        # x = t.max(x, dim=1)[0]
        x = x.view(x.shape[0], -1)

        x = self.embedding_decoder(x)

        # x = x.view(x.shape[0], self.params.in_seq_len, self.params.n_channels, self.params.imsize, self.params.imsize)

        return x
