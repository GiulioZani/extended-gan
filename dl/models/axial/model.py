from dl.base_lightning_modules.base_model import BaseRegressionModel
from ...base_lightning_modules.base_gan_2 import GANLightning

# from ...base_lightning_modules.base_model import BaseRegressionModel

from argparse import Namespace
from ...components.ConvLSTMModule import ConvLSTMClassifier
from ...components.resnetmodel import ResNetFrameDiscriminator
from ...components.conv3dmodel import (
    CompactConv3DDiscriminator,
    Conv3DTemporalDiscriminator,
)
from ...components.lstmconvmodel import ConvLSTMTemporalDiscriminator
from ...components.conv2dmodel import (
    CompactFrameDiscriminator,
    FrameDiscriminator,
    SmallFrameDiscriminator,
)
from ...components.resnet3d import ResNet3DClassifier
from ...components.conv3dmodel import Conv3DFrameDiscriminator

from .axial import AxialGenerator

# from .axial import AxialGenerator


class Model(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = AxialGenerator(params)
        self.temporal_discriminator = ResNet3DClassifier(
            params,
            layers=(3, 3, 3, 3),
            block_inplanes=(4, 8, 8, 16),
        )
        self.frame_discriminator = ResNet3DClassifier(
            params,
            layers=(2, 2, 2, 2),
            block_inplanes=(4, 8, 8, 16),
        )
