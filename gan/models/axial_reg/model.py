from argparse import Namespace
from .base_model import BaseRegressionModel
from .modules import EncoderDecoderConvLSTM
from .axial.axial import AxialGenerator

class Model(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = AxialGenerator(params)
