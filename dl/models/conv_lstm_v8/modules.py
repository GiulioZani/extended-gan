from itertools import chain
import math
from black import out
import ipdb
import torch
import torch.nn as nn
import torch as t
import torch.nn.functional as f

from ...components.ConvLSTMModule import ConvLSTMBlock
from ...components.components import GaussianNoise
from axial_attention import AxialAttention, AxialPositionalEmbedding


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, params, nf=8):
        super(EncoderDecoderConvLSTM, self).__init__()

        self.params = params
        in_chan = 4

        self.conv_lstm_out_chan = 32

        self.z_dim = 16
        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        

        """
        self.act = t.sigmoid

        self.conv_encoders = [
            ConvLSTMBlock(
                in_chan,
                64,
                kernel_size=(3, 3),
                bias=True,
                dropout=False,
                batch_norm=True,
            ),
            ConvLSTMBlock(
                64,
                64,
                kernel_size=(3, 3),
                bias=True,
                dropout=False,
                batch_norm=True,
            ),
       
       
       
        ]


        self.conv_decoder = [
            ConvLSTMBlock(
                64,
                64,
                kernel_size=(3, 3),
                bias=True,
                dropout=0.2,
                batch_norm=True,
            ),
            ConvLSTMBlock(
                64,
                64,
                kernel_size=(3, 3),
                bias=True,
                dropout=False,
                batch_norm=True,
            ),

        ]
        self.conv_lstms = self.conv_encoders + self.conv_decoder

        for i in range(len(self.conv_lstms)):
            setattr(self, "conv_lstm_" + str(i), self.conv_lstms[i])

        self.attention = nn.Sequential(
            AxialAttention(64, num_dimensions=3, heads=4, dim_heads=4),
            nn.LeakyReLU(0.2, inplace=True),
            AxialAttention(64, num_dimensions=3, heads=4, dim_heads=4),
            nn.LeakyReLU(0.2, inplace=True),
        )
 
     
        self.embedding_decoder = nn.Sequential(
            nn.Linear(64, 32, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 4, bias=True),
        )

        self.noise = 0.0001

        self.gaussian_noise = GaussianNoise(self.noise)

    def autoencoder(self, x, seq_len, future_step, h):

        outputs = []

        # ipdb.set_trace()
        # encoder
        for t in range(seq_len):

            input_tensor = x[:, t, :, :, :]
            # ipdb.set_trace()
            # looping over encoders
            for i in range(2):
                input_tensor, c = self.conv_encoders[i](
                    input_tensor=input_tensor, cur_state=h[i]
                )
                h[i] = (input_tensor, c)


        outputs = []
        for t in range(seq_len):


            # looping over encoders
            for i in range(2):
                input_tensor, c = self.conv_decoder[i](
                    input_tensor=input_tensor, cur_state=h[i+2]
                )
                h[i+2] = (input_tensor, c)

            outputs += [input_tensor]
                        
    
        outputs = torch.stack(outputs, 1)

        # ipdb.set_trace()

        outputs = self.attention(outputs)
        
        # outputs = outputs.permute(0, 1, 1, 3, 4).squeeze(2)
        outputs = self.embedding_decoder(outputs)
        # ipdb.set_trace()
        outputs = outputs.permute(0, 1, 4, 2, 3)
        # outputs = outputs.permute(0, 2, 1, 3, 4)
        # outputs = self.decoder_CNN(outputs)
        # outputs = self.decoder_final(outputs)

        # outputs= torch.stack(outputs, 1)
        # ipdb.set_trace()
        # outputs = outputs.view(outputs.shape[0], -1)
        # outputs = self.decoder(outputs)
        return outputs

    def forward(self, x):

        # ipdb.set_trace()



        if self.training:
            x = self.gaussian_noise(x)


        # x = x.unsqueeze(2)
        # ipdb.set_trace()

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # x = x.unsqueeze(2)
        # ipdb.set_trace()

        # find size of different input dimensions

        # noise vector adding to channels

        # x = self.gaussian_noise(x)

        b, seq_len, _, h, w = x.size()

        # ipdb.set_trace()

        # initialize hidden states
        hidden = None
        if hidden == None:
            hidden = []
            for i in range(len(self.conv_lstms)):
                state = self.conv_lstms[i].init_hidden(x)
                hidden += [state]

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, seq_len, hidden)

        # ipdb.set_trace()

        # outputs = outputs.permute(0, 2, 1, 3, 4).squeeze(2)
        # outputs = outputs.squeeze(2)

        # ipdb.set_trace()
        return outputs



