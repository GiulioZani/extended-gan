import torch as t
import torch.nn.functional as F
from torch import nn


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int,
        dropout=0.15,
        nonlinear=True,
        skip=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            chin,
            chout,
            kernel_size,
            padding="same",
        )
        self.expansion = chout / chin
        self.bn = nn.BatchNorm3d(chin)
        self.do = nn.Dropout(dropout)
        self.skip = skip
        self.nonlinear = nonlinear

    def forward(self, x):
        y_hat = self.do(self.conv(x))
        if self.skip:
            if self.expansion > 1:
                x_resized = x.repeat(1, 2, 1, 1, 1)
                y_hat = y_hat.clone() + x_resized
                y_hat = F.relu(y_hat)
            else:
                y_hat = y_hat.clone() + x[:, : y_hat.shape[1]]
                y_hat = F.relu(y_hat)

        return y_hat


class Generator(nn.Module):
    def __init__(self, layer_count:int):
        super().__init__()
        self.layers = nn.Sequential(*[
            ConvBlock2D(4, 16, 4),
            ConvBlock2D(16, 32, 3),
            ConvBlock2D(32, 64, 3),
        ])

    def forward(self, x: t.Tensor):
        


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor):
        pass
