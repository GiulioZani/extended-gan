import threading
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

# from .base_model import BaseModel
import os
import ipdb
from argparse import Namespace
import matplotlib.pyplot as plt
import torchmetrics

from .visualize import _visualize_predictions

# from ..utils.visualize_predictions import visualize_predictions


class CoCycleGAN(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()  # params)
        self.params = params
        self.embedding = nn.Embedding(2, self.params.imsize**2)
        # self.fake_y_detached = t.tensor(0.0)
        # self.fake_x_detached = t.tensor(0.0)
        self.generator = nn.Sequential()
        self.discriminator = nn.Sequential()
        self.best_val_loss = float("inf")
        self.val_mse = torchmetrics.MeanSquaredError()
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        self.train_ssim = torchmetrics.StructuralSimilarityIndexMeasure()

        self.x_future = False
        self.y_future = True
        self.to_visualize = [
            (t.tensor(0), t.tensor(0), t.tensor(0)),
            (t.tensor(0), t.tensor(0), t.tensor(0)),
        ]

    def forward(self, x: t.Tensor):
        return self.generator(x)

    def __embed(self, x: t.Tensor, *, future: bool):
        label = t.tensor(future, device=self.device).int()
        embedded_label = self.embedding(label)
        # .repeat(
        #    self.params.imsize ** 2
        # )
        """
        embedding = (
            embedded_label
            .view(self.params.imsize, self.params.imsize)
            .repeat((x.shape[0], x.shape[1], 1, 1))
            .unsqueeze(2)
        ).to(self.device)
        """
        embedding = embedded_label.view(self.params.imsize, self.params.imsize)
        if len(x.shape) == 4:
            embedding = embedding.repeat((x.shape[0], 1, 1, 1)).to(self.device)
        elif len(x.shape) == 5:
            embedding = embedding.repeat((x.shape[0], x.shape[1], 1, 1, 1)).to(
                self.device
            )
        return t.cat((embedding, x), dim=self.params.channel_dim)

    def embed(self, x: t.Tensor, *, future: bool):
        label = t.tensor(future, device=self.device).int()
        embedded_label = self.embedding(label)
        # .repeat(
        #    self.params.imsize ** 2
        # )
        """
        embedding = (
            embedded_label
            .view(self.params.imsize, self.params.imsize)
            .repeat((x.shape[0], x.shape[1], 1, 1))
            .unsqueeze(2)
        ).to(self.device)
        """
        embedding = embedded_label.view(self.params.imsize, self.params.imsize)
        if len(x.shape) == 4:
            embedding = embedding.repeat((x.shape[0], 1, 1, 1)).to(self.device)
        elif len(x.shape) == 5:
            embedding = embedding.repeat((x.shape[0], x.shape[1], 1, 1, 1)).to(
                self.device
            )
        return t.cat(
            (embedding, x), dim=self.params.channel_dim if len(x.shape) == 5 else 1
        )

    def __unembed(self, x: t.Tensor) -> t.Tensor:
        return x[:, 1:]

    def __visualize_predictions(self, x, y, flag):
        future = self.y_future
        fake_y = self.generator(self.__embed(x, future=future))
        fake_x = self.generator(
            self.__embed(fake_y, future=not future), future=not future
        )

        _visualize_predictions(
            x,
            y,
            fake_y,
            self.generator(self.__embed(y, future=self.x_future), future=self.x_future),
            fake_x,
            self.current_epoch,
            self.params.save_path,
            flag,
        )

    def __save_model(self):
        # run on thread
        thread = threading.Thread(
            target=lambda: t.save(
                self.state_dict(),
                os.path.join(self.params.save_path, "model.pt"),
            ),
            args=(),
        )
        thread.start()

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        generator_loss = self.__generator_loss(x, y, batch_idx, flag="val")
        pred = self.generator(self.__embed(x, future=self.y_future))
        mse_loss = F.mse_loss(pred, y)
        self.val_mse.update(pred.detach(), y.detach())

        if batch_idx == 0:
            self.__visualize_predictions(x, y, "val")

        return {
            "val_loss": generator_loss,
            "val_mse_loss": mse_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()

        avg_mse_loss = self.val_mse.compute()
        sum_mse_loss = t.stack([x["val_mse_loss"] for x in outputs]).sum()
        self.val_mse.reset()

        self.log("val_sum_mse_loss", sum_mse_loss)
        self.log("val_loss", avg_mse_loss, prog_bar=True)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            t.save(
                self.state_dict(),
                os.path.join(self.params.save_path, "best_model.pt"),
            )
            print("Saved model")

        return {"val_loss": avg_mse_loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        generator_loss = self.__generator_loss(x, y, batch_idx, flag="test")
        pred = self.generator(self.__embed(x, future=self.y_future))
        mse_loss = F.mse_loss(pred, y)

        self.val_mse.update(pred.detach(), y.detach())

        if batch_idx == 0:
            self.__visualize_predictions(x, y, "test")
        return {
            "val_loss": generator_loss,
            "val_mse_loss": mse_loss,
        }

    def test_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        # self.log("val_mse_loss", t.stack([x["val_mse_loss"] for x in outputs]).mean())
        # self.log("val_loss", avg_loss, prog_bar=True)
        # self.log("val_gen_loss", avg_loss, prog_bar=True)
        # self.log("val_disc_loss", avg_disc_loss, prog_bar=True)
        sum_mse_loss = t.stack([x["val_mse_loss"] for x in outputs]).sum()
        test_loss = self.val_mse.compute()
        self.log("r_val_mse_loss", test_loss)
        self.log("test_mse", sum_mse_loss)
        self.val_mse.reset()
        return {"val_loss": test_loss}

    def __generator_loss(self, x: t.Tensor, y: t.Tensor, batch_idx: int, flag: str):
        visualize = (batch_idx % 10 == 0) if flag == "train" else (batch_idx == 0)

        # forward_generator_loss = self.__one_way_generator_loss(
        #     x, y, future=self.x_future, visualize=visualize
        # )
        # backward_generator_loss = self.__one_way_generator_loss(
        #     x, y, future=self.y_future, visualize=visualize
        # )

        # cyclic_generator_loss = self.__cyclic_way_generator_loss(x, y)

        if visualize:
            self.__visualize_predictions(x, y, flag)

        pred_y = self.generator(self.__embed(x, future=self.y_future))
        fake_x = self.generator(
            self.__embed(pred_y, future=self.x_future), future=self.x_future
        )
        pred_x = self.generator(
            self.__embed(y, future=self.x_future), future=self.x_future
        )
        cycle_y = self.generator(self.__embed(pred_x, future=self.y_future))

        pred_y_l1 = F.mse_loss(pred_y, y)
        pred_x_l1 = F.mse_loss(pred_x, x)
        pred_cycle_l1 = F.mse_loss(fake_x, x)
        pred_y_cycle_l1 = F.mse_loss(cycle_y, y)

        mse_sum_pred_y_loss = F.mse_loss(pred_y, y, reduction="sum")

        self.log("CoCycleLoss/pred_y_l1", pred_y_l1)

        self.log("CoCycleLoss/pred_y", pred_y_l1)
        self.log("CoCycleLoss/pred_x", pred_x_l1)
        self.log("CoCycleLoss/pred_cycle", pred_cycle_l1)
        self.log("CoCycleLoss/pred_y_cycle", pred_y_cycle_l1)
        self.log("CoCycleLoss/mse_sum_pred_y", mse_sum_pred_y_loss / 10)

        return (
            +self.params.hparams["l1_cyclic_loss_weight"]
            * (pred_cycle_l1 + pred_y_cycle_l1)
            + self.params.hparams["l1_pred_loss_weight"] * pred_y_l1
            + self.params.hparams["l1_past_pred_loss_weight"] * pred_x_l1
        )

    def training_step(
        self,
        batch: tuple[t.Tensor, t.Tensor],
        batch_idx: int,
    ):
        x, y = batch

        if batch_idx % self.params.save_interval == 0:
            # save the model on a thread
            self.__save_model()

        pred = self.generator(self.__embed(x, future=self.y_future))
        mse_loss = F.mse_loss(pred, y)
        self.log("mse_loss", mse_loss, prog_bar=True)
        loss = self.__generator_loss(x, y, batch_idx, flag="train")
        self.log("g_loss", loss, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):

        # avg_loss = t.stack([x["loss"] for x in outputs]).mean()
        # self.log("g_loss", avg_loss, prog_bar=True)
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
            mode="min",
            threshold_mode="rel"
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
