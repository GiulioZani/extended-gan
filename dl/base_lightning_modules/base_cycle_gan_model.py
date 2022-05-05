import torch as t
import torch.nn as nn
import torch.nn.functional as F
# from .base_model import BaseModel
import ipdb
import h5py
# from .utils.data_manager import DataManger
# from .utils.visualize_predictions import visualize_predictions
from argparse import Namespace

from .base_model import BaseRegressionModel


class CycleGANLightning(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.params = params
