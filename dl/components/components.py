import torch as t
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self, variance=0.01):
        super().__init__()
        self.variance = variance

    def forward(self, x, variance=None):
        if variance is None:
            variance = self.variance
        noise = t.randn_like(x) * variance
        x = ((noise + x).detach() - x).detach() + x
        return x
