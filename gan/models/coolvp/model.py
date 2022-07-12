from argparse import Namespace
from .base_model import BaseRegressionModel
from .simvp.model import SimVP

class Model(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = SimVP(params)
