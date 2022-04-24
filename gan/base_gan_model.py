import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import ipdb
import h5py
from .utils.data_manager import DataManger
from .utils.visualize_predictions import visualize_predictions
from argparse import Namespace

class GANLightning(LightningModule):
    def __init__(
        self,
        params: Namespace,
    ):
        super().__init__()

        self.params = params
        # reads file data_location in h5 format
        self.data_manager = DataManger(data_path=params.data_location)
        self.lr = params.lr
        self.b1 = params.b1
        self.b2 = params.b2
        self.generator = nn.Sequential()
        self.frame_discriminator = nn.Sequential()
        self.temporal_discriminator = nn.Sequential()
        self.fake_y_detached = t.tensor(0.0)

    def forward(self, z: t.Tensor) -> t.Tensor:
        return self.generator(z)

    def adversarial_loss(self, y_hat: t.Tensor, y: t.Tensor):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(
        self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int, optimizer_idx: int
    ):
        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape
        # train generator
        if optimizer_idx == 0:
            fake_y = self(x)
            # i save it as property to avoid recomputing it for the discriminators
            self.fake_y_detached = fake_y.detach()  # used later by discriminators
            fake_data_frames = fake_y.reshape(
                batch_size * y_seq_len, channels, height, width
            )
            pred_frame_label = self.frame_discriminator(fake_data_frames)
            pred_temp_label = self.temporal_discriminator(t.cat((x, fake_y), dim=1))
            real_frame_label = t.ones(batch_size * y_seq_len).to(self.device)
            real_temp_label = t.ones(batch_size).to(self.device)
            generator_loss = self.adversarial_loss(
                pred_temp_label, real_temp_label
            ) + self.adversarial_loss(pred_frame_label, real_frame_label)
            tqdm_dict = {"g_loss": generator_loss}
            # not used for backpropagation
            train_mse = F.mse_loss(fake_y.detach(), y)
            return {
                "loss": generator_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "train_mse": train_mse,
            }

        # train frame discriminator
        if optimizer_idx == 1:
            # Create labels for the real data. (label=1)
            # x_frames = x.view(batch_size * x_seq_len, channels, height, width)
            y_frames = y.reshape(batch_size * y_seq_len, channels, height, width)
            fake_frames = self.fake_y_detached.reshape(
                batch_size * y_seq_len, channels, height, width
            )
            y_batch_size = batch_size * y_seq_len
            real_label = t.ones(y_batch_size, 1, device=self.device)
            fake_label = t.zeros(y_batch_size, 1, device=self.device)
            labels = t.cat((real_label, fake_label)).squeeze()
            frames = t.cat((y_frames, fake_frames))
            frame_disc_loss = self.adversarial_loss(
                self.frame_discriminator(frames).squeeze(), labels
            )
            tqdm_dict = {"fd_loss": frame_disc_loss}
            return {
                "loss": frame_disc_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }

        # train temporal discriminator
        if optimizer_idx == 2:
            real_label = t.ones(batch_size, 1, device=self.device)
            fake_label = t.zeros(batch_size, 1, device=self.device)
            labels = t.cat((real_label, fake_label)).squeeze()
            fake_sequence = t.cat((x, self.fake_y_detached), dim=1)
            real_sequence = t.cat((x, y), dim=1)
            sequences = t.cat((real_sequence, fake_sequence))
            pred_labels = self.temporal_discriminator(sequences)
            temp_disc_loss = self.adversarial_loss(pred_labels, labels)
            tqdm_dict = {"td_loss": temp_disc_loss}
            return {
                "loss": temp_disc_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x[0]["train_mse"] for x in outputs]).mean()
        self.log("train_mse", avg_loss, prog_bar=True)

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(x, y, self(x), path=self.params.save_path)

        pred_y = self(x)
        loss = F.mse_loss(pred_y, y)
        self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(x, y, self(x), path=self.params.save_path)

        pred_y = self(x)
        se = F.mse_loss(pred_y, y, reduction="sum")
        denorm_pred_y = self.data_manager.denormalize(pred_y, self.device)
        denorm_y = self.data_manager.denormalize(y, self.device)
        ae = F.l1_loss(denorm_pred_y, denorm_y, reduction="sum")
        mask_pred_y = self.data_manager.discretize(denorm_pred_y, self.device)
        mask_y = self.data_manager.discretize(denorm_y, self.device)
        tn, fp, fn, tp = t.bincount(
            mask_y.flatten()*2 + mask_pred_y.flatten(), minlength=4,
        )
        total_lengh = mask_y.numel()
        return {
            "se": se,
            "ae": ae,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_lengh": total_lengh,
        }

    def test_epoch_end(self, outputs):
        total_lenght = sum([x["total_lengh"] for x in outputs])
        mse = (t.stack([x["se"] for x in outputs]).sum() / total_lenght)
        mae = (t.stack([x["ae"] for x in outputs]).sum() / total_lenght)
        tn = (t.stack([x["tn"] for x in outputs]).sum() / total_lenght)
        fp = (t.stack([x["fp"] for x in outputs]).sum() / total_lenght)
        fn = (t.stack([x["fn"] for x in outputs]).sum() / total_lenght)
        tp = (t.stack([x["tp"] for x in outputs]).sum() / total_lenght)
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
        """
        self.log("test_mse", mse.item())
        self.log("test_mae", mae.item())
        self.log("test_accuracy", accuracy.item(), prog_bar=True)
        self.log("test_precision", precision.item(), prog_bar=True)
        self.log("test_recall", recall.item(), prog_bar=True)
        self.log("test_f1", f1.item(), prog_bar=True)
        """
        self.log("test_performance", test_metrics, prog_bar=True)
        

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        frame_discriminator_optimizer = t.optim.Adam(
            self.frame_discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        temporal_discriminator_optimizer = t.optim.Adam(
            self.temporal_discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return (
            [
                generator_optimizer,
                frame_discriminator_optimizer,
                temporal_discriminator_optimizer,
            ],
            [],
        )
