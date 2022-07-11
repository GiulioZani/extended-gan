from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace
from torchmetrics import Accuracy
import torchmetrics

from .visualize import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os
import mate


class BaseRegressionModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        loss = t.nn.MSELoss()
        self.loss = lambda x, y: loss(x.flatten(), y.flatten())  # t.nn.MSELoss()
        self.mse_metric = torchmetrics.MeanSquaredError()
        self.ssim = mate.metrics.MovingSSIM()
        self.sum_mse = t.nn.MSELoss(reduction="sum")

        # self.denorm_mse = mate.DenormMSE()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        if batch_idx % self.params.save_interval == 0:
            visualize_predictions(
                x,
                y,
                y_pred,
                path=self.params.save_path,
                epoch=self.current_epoch,
                label="train",
            )

        sum_mse = F.mse_loss(y_pred, y, reduction="sum")
        self.log("sum_mse", sum_mse, prog_bar=True)
        return {"loss": loss, "sum_mse": sum_mse}

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        avg_sum_mse = t.stack([x["val_sum_mse"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_sum_mse", avg_sum_mse, prog_bar=True)


        return {"val_mse": avg_loss}

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(
                x,
                y,
                self(x),
                self.current_epoch,
                path=self.params.save_path,
                label="val",
            )
        y = y.cpu()
        pred_y = self(x).cpu()
        loss = F.mse_loss(pred_y, y)
        sum_mse = F.mse_loss(pred_y, y, reduction="sum")
        return {"val_mse": loss, "val_loss": loss, "val_sum_mse": sum_mse}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(
                x,
                y,
                self(x),
                path=self.params.save_path,
                epoch=self.current_epoch,
                label="test",
            )

        pred_y = self(x)

        self.ssim.update(y, pred_y)

        # se = F.mse_loss(pred_y, y, reduction="sum")
        self.mse_metric.update(y, pred_y)
        sum_mse = F.mse_loss(pred_y, y, reduction="sum")
        return {"test_mse": self.mse_metric.compute(), "test_sum_mse": sum_mse}

    def test_epoch_end(self, outputs):
        avg_test_mse = t.stack([x["test_sum_mse"] for x in outputs]).mean()
        loss = self.mse_metric.compute()
        self.log("test_mse", loss, prog_bar=True)
        self.mse_metric.reset()
        self.log("test_sum_mse", avg_test_mse, prog_bar=True)

        ssim = self.ssim.compute()
        self.log("test_ssim", ssim, prog_bar=True)
        self.ssim.reset()

        return {"test_mse": loss}

    def configure_optimizers(self):
        # lr = self.params.lr
        # b1 = self.params.b1
        # b2 = self.params.b2

        # generator_optimizer = t.optim.Adam(
        #     self.generator.parameters(), lr=lr, betas=(b1, b2)
        # )

        # generator_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
        #     generator_optimizer,
        #     patience=self.params.reduce_lr_on_plateau_patience,
        #     verbose=True,
        #     factor=self.params.reduce_lr_on_plateau_factor,
        #     threshold=self.params.reduce_lr_on_plateau_threshold,
        #     threshold_mode="rel",
        #     mode="min",
        # )

        optimizer = mate.Optimizer(self.params.optimizer, self.generator)

        return optimizer.get_optimizer()
