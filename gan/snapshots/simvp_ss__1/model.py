from argparse import Namespace
from .base_model import BaseGanLightning
from .convlstm.ConvLSTMModule import ConvLSTMClassifier

from .conv2d.conv2dmodel import FrameDiscriminator
from .resnet.resnet3d import ResNet3DClassifier
from .resnet.resnetmodel import ResNetFrameDiscriminator, ResNetTemproalDiscriminator
from .simvp.model import SimVP, SimVPTemporalDiscriminator
from .axial.axial import AxialDiscriminator


class Model(BaseGanLightning):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = SimVP(params)
        self.frame_discriminator = ResNetFrameDiscriminator(params)
        self.temporal_discriminator = ResNet3DClassifier(
            params,  block_inplanes=(32, 64, 128, 256)
        )
        self.second_temporal_discriminator = ResNet3DClassifier(
            params, block_inplanes=(32, 64, 128, 256)
        )
