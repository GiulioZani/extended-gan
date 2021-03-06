import torch as t
import torch.nn as nn
import torch.nn.functional as F
import ipdb


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find("conv") != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find("bn") != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class ConvBlock(nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int,
        *,
        bias=True,
        stride=1,
        padding=0,
        dropout=0.01,
        act=F.relu,
        batchnorm=True
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                chin,
                chout,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(chout))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.act = act
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.act(self.layers(x))


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Input is the latent vector Z.
        self.layers = nn.Sequential(
            ConvBlock(params["nc"], params["nc"] * 8, kernel_size=4, padding="same"),
            ConvBlock(params["nc"] * 8, params["nc"] * 4, 4, padding="same"),
            ConvBlock(params["nc"] * 4, params["nc"] * 2, 4, padding="same"),
            ConvBlock(params["nc"] * 2, params["nc"], 4, padding="same"),
            ConvBlock(
                params["nc"],
                params["nc"],
                4,
                padding="same",
                act=t.sigmoid,
                batchnorm=False,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class TemporalDiscriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        nc = params["nc"]
        ndf = params["ndf"]

        def act(x):
            return F.leaky_relu(x, 0.2, True)

        self.layers = nn.Sequential(
            *[
                ConvBlock(
                    2 * nc,
                    ndf,
                    kernel_size=4,
                    stride=2,
                    bias=False,
                    batchnorm=False,
                    padding=1,
                    act=act,
                ),
                ConvBlock(
                    ndf,
                    2 * ndf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    act=act,
                ),
                ConvBlock(
                    2 * ndf,
                    4 * ndf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    act=act,
                ),
                ConvBlock(
                    4 * ndf,
                    8 * ndf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    act=act,
                ),
                ConvBlock(
                    8 * ndf,
                    1,
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=False,
                    batchnorm=False,
                    act=t.sigmoid,
                ),
            ]
        )

    def forward(self, x):
        x = self.layers(x)
        return x.squeeze()


class FrameDiscriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params["nc"], params["ndf"], 4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params["ndf"], params["ndf"] * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params["ndf"] * 2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(
            params["ndf"] * 2, params["ndf"] * 4, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(params["ndf"] * 4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(
            params["ndf"] * 4, params["ndf"] * 8, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(params["ndf"] * 8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params["ndf"] * 8, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = t.sigmoid(self.conv5(x))

        return x.squeeze()
