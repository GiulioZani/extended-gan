import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import ipdb
import h5py
from .utils.data_manager import DataManger


class GANLightning(LightningModule):
    def __init__(
        self,
        data_location: str,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        # reads file data_location in h5 format
        self.data_manager = DataManger(data_path=data_location)
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
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
            fake_data_frames = fake_y.view(
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
            y_frames = y.view(batch_size * y_seq_len, channels, height, width)
            fake_frames = self.fake_y_detached.view(
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
        pred_y = self(x)
        loss = F.mse_loss(pred_y, y)
        self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        pred_y = self(x)
        se = F.mse_loss(pred_y, y, reduction="sum")
        denorm_pred_y = self.data_manager.denormalize(pred_y)
        denorm_y = self.data_manager.denormalize(y)
        ae = F.l1_loss(denorm_pred_y, denorm_y, reduction="sum")
        mask_pred_y = self.data_manager.discretize(denorm_pred_y)
        mask_y = self.data_manager.discretize(denorm_y)
        tn, fp, fn, tp = t.bincount(
            mask_y.flatten() + mask_pred_y.flatten(), minlength=4,
        )
        total_lengh = mask_y.numel()
        return {"se": se, "ae": ae, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "total_lengh": total_lengh}

    def test_epoch_end(self, outputs):
        total_lenght = t.stack([x["total_lengh"] for x in outputs]).sum()
        mse = t.stack([x["se"] for x in outputs])/total_lenght
        mae = t.stack([x["ae"] for x in outputs])/total_lenght
        tn = t.stack([x["tn"] for x in outputs])/total_lenght
        fp = t.stack([x["fp"] for x in outputs])/total_lenght
        fn = t.stack([x["fn"] for x in outputs])/total_lenght
        tp = t.stack([x["tp"] for x in outputs])/total_lenght
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        self.log("test_mse", mse, prog_bar=True)
        self.log("test_mae", mae, prog_bar=True)
        self.log("test_accuracy", tn, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

        
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
