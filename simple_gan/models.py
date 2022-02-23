import torch as t
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor):
        pass


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: t.Tensor):
        pass
