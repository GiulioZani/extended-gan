import math
from turtle import forward
import torch as t
import torch.nn as nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb

from ..axial_autoencoder_embedding.modules import AutoEncoder
from ...components.components import GaussianNoise
from torch.functional import F


class AxialLayers(nn.Module):
    def __init__(
        self,
        embedding_dim=8,
        num_layers=4,
        embedding_idx=2,
        num_heads=8,
        dropout=0.00,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_idx = embedding_idx
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.attentions = nn.Sequential()
        for i in range(self.num_layers):
            # we need setattr so that lightning can find the module
            self.__setattr__(
                "layer_{}".format(i),
                AxialAttention(self.embedding_dim, 2, self.num_heads, 4, 1, True),
            )
            self.attentions.add_module(
                "layer_{}".format(i), self.__getattr__("layer_{}".format(i))
            )
        self.do = nn.Dropout(self.dropout)

    def forward(self, x):
        z = self.attentions(x)
        out = z  # + x
        out = F.relu(out)
        out = self.do(out)
        return out


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

        # set no grad for embedding encoder and decoder
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
            # nn.LeakyReLU(0.2),
            # nn.Dropout(0.10),
            AxialAttention(
                dim=self.embedding_dim,  # embedding dimension
                dim_index=-1,  # where is the embedding dimension
                heads=8,  # number of heads for multi-head attention,
                dim_heads=4,
                num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
            ),
            # nn.Dropout(0.10),
            # nn.LeakyReLU(0.2),
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
