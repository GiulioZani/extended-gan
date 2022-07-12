import threading
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

import os
import ipdb
from argparse import Namespace
import matplotlib.pyplot as plt
import torchmetrics

from .visualize import _visualize_predictions
import mate


class CoCycleGAN(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()  # params)
        self.params = params

        self.generator = nn.Sequential()
        self.discriminator = nn.Sequential()
        self.val_mse = torchmetrics.MeanSquaredError()

        self.x_future = False
        self.y_future = True
        self.to_visualize = [
            (t.tensor(0), t.tensor(0), t.tensor(0)),
            (t.tensor(0), t.tensor(0), t.tensor(0)),
        ]

        self.__losses = {
            "l1": F.l1_loss,
            "mse": F.mse_loss,
            "l1_smooth": F.smooth_l1_loss,
        }

        # self.denorm_mse = mate.metrics.DenormMSE(lambda x: (x + 1) / 2)

        self.__cycle_loss = self.__losses[self.params.hparams.cycle_similarity_function]
        self.__loss_function = self.__losses[self.params.hparams.pred_loss_function]

    def forward(self, x: t.Tensor, future: bool = False) -> t.Tensor:
        return self.generator(x, future=future)

    def __visualize_predictions(self, x, y, flag):
        future = self.y_future
        fake_y = self.generator(x, future=future)
        fake_x = self.generator(fake_y, future=not future)

        _visualize_predictions(
            x,
            y,
            fake_y,
            self.generator(y, future=self.x_future),
            fake_x,
            self.current_epoch,
            self.params.save_path,
            flag,
        )

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        # generator_loss = self.__generator_loss(x, y, batch_idx, flag="val")
        pred = self.generator(x, future=self.y_future)
        mse_loss = F.mse_loss(pred, y)
        self.val_mse.update(pred.detach(), y.detach())

        mse_sum = t.mean((pred - y) ** 2, axis=(0, 1, 2)).sum()

        if batch_idx == 0:
            self.__visualize_predictions(x, y, "val")

        return {
            "val_loss": mse_loss,
            "mse_sum": mse_sum,
            # "val_mse_loss": mse_loss,
        }

    def validation_epoch_end(self, outputs):

        avg_mse_loss = self.val_mse.compute()
        sum_mse_loss = t.stack([x["mse_sum"] for x in outputs]).sum()

        self.val_mse.reset()

        self.log("val_loss", avg_mse_loss, prog_bar=True)
        self.log("val_mse_sum", sum_mse_loss, prog_bar=True)

        return {"val_loss": avg_mse_loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        # generator_loss = self.__generator_loss(x, y, batch_idx, flag="test")
        pred = self.generator(x, future=self.y_future)
        mse_sum = t.mean((pred - y) ** 2, axis=(0, 1, 2)).sum()

        self.val_mse.update(pred.detach(), y.detach())

        if batch_idx == 0:
            self.__visualize_predictions(x, y, "test")
        return {"mse_loss": mse_sum}

    def test_epoch_end(self, outputs):

        test_loss = self.val_mse.compute()
        mse_avg_loss = t.stack([x["mse_loss"] for x in outputs]).mean()
        self.log("test_loss", test_loss, prog_bar=True)

        self.val_mse.reset()
        return {"test_loss": test_loss, "test_mse_loss": mse_avg_loss}

    def training_step(
        self,
        batch: tuple[t.Tensor, t.Tensor],
        batch_idx: int,
    ):
        x, y = batch

        if batch_idx % self.params.save_interval == 0:
            self.__visualize_predictions(x, y, "train")

        pred_y = self.generator(x, future=self.y_future)
        pred_x = self.generator(y, future=self.x_future)

        pred_y_loss = self.__loss_function(pred_y, y)
        pred_x_loss = self.__loss_function(pred_x, x)

        loss = self.params.hparams.future_pred_loss_weight * pred_y_loss

        # self.denorm_mse.update(pred_y, y)
        # self.log("denorm_mse", self.denorm_mse.compute(), prog_bar=True)

        if self.params.hparams.past_pred_loss_weight > 0:
            loss += self.params.hparams.past_pred_loss_weight * pred_x_loss

        if self.params.hparams.cycle_loss_weight > 0:
            cycle_x = self.generator(pred_y, future=self.x_future)
            cycle_y = self.generator(pred_x, future=self.y_future)

            cycle_x_loss = self.__cycle_loss(cycle_x, x)
            cycle_y_loss = self.__cycle_loss(cycle_y, y)

            loss += self.params.hparams.cycle_loss_weight * (
                cycle_x_loss + cycle_y_loss
            )

        # mse_sum = F.mse_loss(pred_y, y, reduction="sum")
        # self.log("t_mse_sum", mse_sum, prog_bar=True)
        mse_batch = t.mean((pred_y - y) ** 2, axis=(0, 1, 2)).sum()
        # ipdb.set_trace()
        self.log("pred_loss", pred_y_loss, prog_bar=True)
        self.log("mse_loss", mse_batch, prog_bar=True)

        return {"loss": loss, "mse_batch": mse_batch}

    def training_epoch_end(self, outputs):

        avg_loss = t.stack([x["mse_batch"] for x in outputs]).mean()
        # self.log("avg_l", avg_loss, prog_bar=True)
        # return {"g_loss": avg_loss}
        return None

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )

        # initializes reduce_lr_on_plateau scheduler
        generator_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer,
            patience=self.params.reduce_lr_on_plateau_patience,
            verbose=True,
            factor=self.params.reduce_lr_on_plateau_factor,
        )
        """
        (
            [generator_optimizer, temporal_discriminator_optimizer,],
            [],
        )
        """
        return [
            {
                "optimizer": generator_optimizer,
                "lr_scheduler": {
                    "scheduler": generator_scheduler,
                    "monitor": "val_loss",
                },
            },
            # {
            #     "optimizer": discriminator_optimizer,
            #     "lr_scheduler": {
            #         "scheduler": discriminator_scheduler,
            #         "monitor": "val_loss",
            #     },
            # },
        ]
