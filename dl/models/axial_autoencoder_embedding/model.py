from ...base_lightning_modules.base_autoencoder_model import BaseAutoEncoderModel

from argparse import Namespace
from .modules import AutoEncoder

class Model(BaseAutoEncoderModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = AutoEncoder(128)
