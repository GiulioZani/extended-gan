from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

from ..utils.visualize_predictions import visualize_predictions
from ..utils.data_manager import DataManger
import matplotlib.pyplot as plt
import ipdb
import os


class BaseAutoEncoderModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.data_manager = DataManger(data_path=params.data_location)
        self.generator = t.nn.Sequential()
        loss = t.nn.BCELoss()
        self.loss = lambda x, y: loss(x.flatten(), y.flatten())  # t.nn.MSELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        x = t.cat([x, y], dim=1)        
        y = x
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        t.save(
            self.state_dict(),
            os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        return {"val_mse": avg_loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        # if batch_idx == 0:
        #     visualize_predictions(x, y, self(x), path=self.params.save_path)
        # y = y.cpu()
        x = t.cat([x, y], dim=1)
        y = x

        pred_y = self(x)
        loss = F.mse_loss(pred_y, y)
        # self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss, "val_loss": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        x = t.cat([x, y], dim=1)
        y = x
        pred_y = self(x)
        se = F.mse_loss(pred_y, y, reduction="sum")
        return {
            "test_loss": se,
        }

    def test_epoch_end(self, outputs):
        total_lenght = sum([x["test_loss"] for x in outputs])

        avg_loss = total_lenght / len(outputs)
        self.log("test_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )

        return generator_optimizer
