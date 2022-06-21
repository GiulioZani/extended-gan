from argparse import Namespace
from .base_model import BaseRegressionModel
from .modules import EncoderDecoderConvLSTM


class Model(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = EncoderDecoderConvLSTM(params)
