from ...base_lightning_modules.cocycle_gan_model import CoCycleGAN
from .old_model import EncoderDecoderConvLSTM
from ...components.resnet3d import (
    ResNet3DAutoEncoder,
    ResNet3DClassifier,
)
from argparse import Namespace


class Model(CoCycleGAN):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)
        self.temporal_discriminator = ResNet3DClassifier(params)
