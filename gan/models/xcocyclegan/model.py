from argparse import Namespace
from .cocycle_gan_model import CoCycleGAN
from .convlstm.ConvLSTMModule import ConvLSTMClassifier

from .conv2d.conv2dmodel import FrameDiscriminator
from .resnet.resnet3d import ResNet3DClassifier
from .resnet.resnetmodel import ResNetFrameDiscriminator, ResNetTemproalDiscriminator
from .simvp.model import SimVP, SimVPTemporalDiscriminator
from .axial.axial import AxialDiscriminator


class Model(CoCycleGAN):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = SimVP(params)
        # self.frame_discriminator = ResNetFrameDiscriminator(params)
        self.discriminator = ResNetTemproalDiscriminator(
            params, [4, 4, 4], [32, 64, 128]
        )
        # self.discriminator = ResNet3DClassifier(
        #     params, block_inplanes=(32, 64, 128, 256)
        # )
