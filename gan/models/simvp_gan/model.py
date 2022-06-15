from argparse import Namespace
from .base_model import BaseGanLightning
from .modules import EncoderDecoderConvLSTM

from .conv2d.conv2dmodel import FrameDiscriminator
from .resnet.resnet3d import ResNet3DClassifier
from .resnet.resnetmodel import ResNetFrameDiscriminator
from .simvp.model import SimVP, SimVPTemporalDiscriminator

class Model(BaseGanLightning):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = SimVP(params)
        self.frame_discriminator = ResNetFrameDiscriminator(params)
        self.temporal_discriminator = SimVPTemporalDiscriminator(params)
