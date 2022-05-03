from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace

# from ..utils.visualize_predictions import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os


class BaseRegressionModel(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()
        # self.data_manager = DataManger(data_path=params.data_location)
        self.generator = t.nn.Sequential()
        loss = t.nn.BCELoss()
        self.loss = lambda x, y: loss(
            x.flatten(), y.flatten()
        )  # t.nn.MSELoss()

    def forward(self, z: t.Tensor) -> t.Tensor:
        out = self.generator(z)
        return out

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        t.save(
            self.state_dict(),
            os.path.join(self.params.save_path, "checkpoint.ckpt"),
        )
        return {"val_mse": avg_loss}

    def validation_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int
    ):
        x, y = batch
        y_pred = self(x)
        if batch_idx == 0:
            # visualize_predictions(x, y, self(x), path=self.params.save_path)
            pass
        loss = F.mse_loss(y_pred, y)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = t.optim.Adam(
            self.parameters(), lr=self.params.lr,  # weight_decay=0.01
        )
        scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.params.reduce_lr_on_plateau_patience,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mse",
        }

    def plot_predictions(
        self, x: t.Tensor, y: t.Tensor, pred_y: t.Tensor, title: str
    ):
        pass

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        y_pred = self(x)
        if batch_idx == 0:
            self.plot_predictions(x, y, y_pred, "test")

        return self.test_without_forward(y, y_pred)

    def test_without_forward(self, y, pred_y):
        y_single = y  # [:, :, -1, 2]
        pred_y_single = pred_y  # [:, :, -1, 2]
        se = F.mse_loss(pred_y_single, y_single, reduction="sum")
        denorm_pred_y = self.params.data_manager.denormalize(
            pred_y
        )  # , self.device)
        denorm_y = self.params.data_manager.denormalize(y)  # , self.device)
        ae = F.l1_loss(
            denorm_pred_y,
            denorm_y,
            reduction="sum",  # [:, :, -1, 2],  # [:, :, -1, 2],
        )
        mask_pred_y = self.params.data_manager.discretize(
            denorm_pred_y
        )  # , self.device)
        mask_y = self.params.data_manager.discretize(denorm_y)  # [:, :, -1, 2]
        tn, fp, fn, tp = t.bincount(
            mask_y.flatten() * 2 + mask_pred_y.flatten(), minlength=4,
        )
        total_lengh = mask_y.numel()
        return {
            "se": se,
            "ae": ae,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_length": total_lengh,
        }

    def test_epoch_end(self, outputs):
        total_lenght = sum([x["total_length"] for x in outputs])
        mse = t.stack([x["se"] for x in outputs]).sum() / total_lenght
        mae = t.stack([x["ae"] for x in outputs]).sum() / total_lenght
        tn = t.stack([x["tn"] for x in outputs]).sum() / total_lenght
        fp = t.stack([x["fp"] for x in outputs]).sum() / total_lenght
        fn = t.stack([x["fn"] for x in outputs]).sum() / total_lenght
        tp = t.stack([x["tp"] for x in outputs]).sum() / total_lenght
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        test_metrics = {
            "mse": mse,
            "mae": mae,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
        }
        test_metrics = {k: v for k, v in test_metrics.items()}
        self.log("test_performance", test_metrics, prog_bar=True)
