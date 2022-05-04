from dl.base_lightning_modules.base_model import BaseRegressionModel

from argparse import Namespace
from .modules import AxialGenerator

class Model(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = AxialGenerator(params)
