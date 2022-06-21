from itertools import chain
import math
from black import out
import ipdb
import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as f
from .convlstm.ConvLSTMModule import ConvLSTMBlock
from .conv2d.conv2dmodel import ConvBlock


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params, nf=8):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model
        """

        self.act = t.tanh
        self.params = params
        in_chan = params.n_channels

        self.conv2d_encoder = nn.Sequential(
            ConvBlock(
                in_chan,
                64,
                kernel_size=3,
                padding="same",
                bias=True,
                dropout=0.1,
                batchnorm=True,
            ),
            ConvBlock(
                64,
                128,
                kernel_size=4,
                padding="same",
                bias=True,
                dropout=0.1,
                batchnorm=True,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(
                128,
                256,
                kernel_size=4,
                padding="same",
                bias=True,
                dropout=0.1,
                batchnorm=True,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(
                256,
                512,
                kernel_size=4,
                padding="same",
                bias=True,
                dropout=0.1,
                batchnorm=True,
            ),
        )

        self.conv_encoders = [
            ConvLSTMBlock(
                512,
                512,
                kernel_size=(3, 3),
                bias=True,
                dropout=0.1,
                batch_norm=True,
            ),
            ConvLSTMBlock(
                512,
                512,
                kernel_size=(3, 3),
                bias=True,
                dropout=0.1,
                batch_norm=True,
            ),
        ]

        self.conv_decoder = [
            ConvLSTMBlock(
                512,
                512,
                kernel_size=(3, 3),
                bias=True,
                dropout=0.1,
                batch_norm=True,
            ),
            ConvLSTMBlock(
                512,
                512,
                kernel_size=(3, 3),
                bias=True,
                dropout=False,
                batch_norm=True,
            ),
        ]
        self.conv_lstms = self.conv_encoders + self.conv_decoder

        for i in range(len(self.conv_lstms)):
            setattr(self, "conv_lstm_" + str(i), self.conv_lstms[i])

        self.decoder_CNN = nn.Sequential(
            ConvBlock(512, 256, 3, padding="same", dropout=0.1, batchnorm=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ConvBlock(256, 128, 3, padding="same", dropout=0.1, batchnorm=True),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            ConvBlock(128, 64, 3, padding="same", dropout=0.1, batchnorm=True),
            ConvBlock(64, 1, 3, padding="same", dropout=0.0, batchnorm=True),
            nn.Tanh(),
        )

        self.noise = 0.0001

    def autoencoder(self, x, seq_len, future_step, h):

        outputs = []

        # ipdb.set_trace()

        # encoder
        for t in range(seq_len):

            input_tensor = self.conv2d_encoder(x[:, t, :, :, :])
            # ipdb.set_trace()
            # looping over encoders
            for i in range(2):
                input_tensor, c = self.conv_encoders[i](
                    input_tensor=input_tensor, cur_state=h[i]
                )
                h[i] = (input_tensor, c)

        input_tensor = h[1][0]
        outputs = []
        for t in range(future_step):

            # input_tensor = x[:, t, :, :, :]
            # looping over encoders
            for i in range(2, 4):
                input_tensor, c = self.conv_decoder[i - 2](
                    input_tensor=input_tensor, cur_state=h[i]
                )
                h[i] = (input_tensor, c)

            # ipdb.set_trace()
            outputs += [self.decoder_CNN(input_tensor)]

        # for t in range(future_step):

        #     for i in range(len(self.conv_decoder)):
        #         input_tensor, c = self.conv_decoder[i](
        #             input_tensor=input_tensor, cur_state=h[i + 4]
        #         )
        #         h[i + 4] = (input_tensor, c)

        #     outputs += [input_tensor]

        outputs = torch.stack(outputs, 1)
        # ipdb.set_trace()
        # outputs = outputs.permute(0, 2, 1, 3, 4)
        # outputs = self.decoder_CNN(outputs)
        # outputs = self.decoder_final(outputs)

        # outputs= torch.stack(outputs, 1)
        # # ipdb.set_trace()
        # outputs = outputs.view(outputs.shape[0], -1)
        # outputs = self.decoder(outputs)
        return outputs

    def forward(self, x):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        b, seq_len, _, h, w = x.size()

        hidden = None
        if hidden == None:
            hidden = []
            for i in range(len(self.conv_lstms)):
                shape = (b, h // 4, w // 4)
                # ipdb.set_trace()
                state = self.conv_lstms[i].init_hidden(shape)
                hidden += [state]

        # autoencoder forward

        outputs = self.autoencoder(x, seq_len, seq_len, hidden)
        # ipdb.set_trace()
        # outputs = outputs.permute(0, 2, 1, 3, 4).squeeze(2)

        return outputs
