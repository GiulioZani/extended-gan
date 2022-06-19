from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace
from torchmetrics import Accuracy

from .visualize import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os
from .sam import SAM


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, t.nn.BatchNorm2d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, t.nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class BaseRegressionModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        self.generator = t.nn.Sequential()
        self.automatic_optimization = False
        loss = t.nn.MSELoss()
        self.loss = lambda x, y: loss(x.flatten(), y.flatten())  # t.nn.MSELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        # ipdb.set_trace()

        optimizer = self.optimizers()

        x, y = batch
        enable_running_stats(self.generator)
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.manual_backward(loss)
        optimizer.first_step(zero_grad=True)

        if batch_idx % 100 == 0:
            visualize_predictions(x, y, y_pred, path=self.params.save_path)

        disable_running_stats(self.generator)
        loss_2 = self.loss(self(x), y)
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)

        self.log("loss", loss, prog_bar=True)
        self.log("loss_2", loss_2, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        t.save(
            self.state_dict(),
            os.path.join(self.params.save_path, "model.pt"),
        )

        self.lr_scheduler_step(self.lr_schedulers(), 0, avg_loss)

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
        return {"val_loss": loss}

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

        base_optimizer = t.optim.SGD
        generator_optimizer = SAM(
            self.generator.parameters(), t.optim.SGD, lr=0.0001, momentum=0.9
        )

        generator_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer,
            patience=self.params.reduce_lr_on_plateau_patience,
            verbose=True,
            factor=self.params.reduce_lr_on_plateau_factor,
        )

        # return generator_optimizer

        return {
            "optimizer": generator_optimizer,
            "lr_scheduler": {
                "scheduler": generator_scheduler,
                "monitor": "loss",
            },
        }
