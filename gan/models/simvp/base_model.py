from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace
from torchmetrics import Accuracy

from .visualize import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os


class BaseRegressionModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        loss = t.nn.MSELoss()
        self.loss = lambda x, y: loss(x.flatten(), y.flatten())  # t.nn.MSELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y) + F.l1_loss(y_pred, y)

        if batch_idx % 100 == 0:
            visualize_predictions(x, y, y_pred, path=self.params.save_path)

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
        if batch_idx == 0:
            visualize_predictions(x, y, self(x), path=self.params.save_path)
        y = y.cpu()
        pred_y = self(x).cpu()
        loss = F.mse_loss(pred_y, y)
        # self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss, "val_loss": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(x, y, self(x), path=self.params.save_path)

        pred_y = self(x)
        se = F.mse_loss(pred_y, y, reduction="sum")

    def test_epoch_end(self, outputs):

        return {}

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )

        generator_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer,
            patience=self.params.reduce_lr_on_plateau_patience,
            verbose=True,
            factor=self.params.reduce_lr_on_plateau_factor,
        )

        return {
            "optimizer": generator_optimizer,
            "lr_scheduler": {
                "scheduler": generator_scheduler,
                "monitor": "loss",
            },
        }
