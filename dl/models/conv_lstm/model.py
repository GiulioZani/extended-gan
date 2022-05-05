from ...base_lightning_modules.base_gan_2 import GANLightning
from ...components.ConvLSTMModule import ConvLSTMClassifier
from .modules import EncoderDecoderConvLSTM
from ...components.resnet3d import (
    ResNet3DAutoEncoder,
    ResNet3DClassifier,
)
from ...components.conv2dmodel import FrameDiscriminator, SmallFrameDiscriminator
from argparse import Namespace


class Model(GANLightning):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)
        self.temporal_discriminator = ResNet3DClassifier(params, seq_len=20)
        self.frame_discriminator = ResNet3DClassifier(params, seq_len=10)
