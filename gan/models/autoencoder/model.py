from argparse import Namespace

from .base_model import BaseAutoEncoder
from .conv2d.conv2dmodel import Conv2DAutoEncoder
from .linear.linearmodel import LinearAutoEncoder

class Model(BaseAutoEncoder):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.generator = LinearAutoEncoder(params)
