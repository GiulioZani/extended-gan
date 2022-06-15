from argparse import Namespace
from .base_model import BaseGanLightning
from .modules import EncoderDecoderConvLSTM
from .axial.axial import AxialGenerator, AxialDiscriminator, AxialFrameDiscriminator

from .conv2d.conv2dmodel import FrameDiscriminator, SmallFrameDiscriminator
from .resnet.resnet3d import ResNet3DClassifier
from .resnet.resnetmodel import ResNetFrameDiscriminator

class Model(BaseGanLightning):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = AxialGenerator(params)
        self.frame_discriminator = ResNetFrameDiscriminator(params)
        self.temporal_discriminator = ResNet3DClassifier(params, block_inplanes=(4, 8, 16, 64))
