from turtle import forward
from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace
from torchmetrics import Accuracy
import torch.nn as nn
from .visualize import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os


class BaseGanLightning(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()

        self.generator = nn.Sequential()
        self.frame_discriminator = nn.Sequential()
        self.temporal_discriminator = nn.Sequential()
        self.fake_y_detached = t.tensor(0.0)
        self.l1_loss = nn.L1Loss()

    def adversarial_loss(self, y_hat: t.Tensor, y: t.Tensor):
        return F.binary_cross_entropy(y_hat.flatten(), y.flatten()).mean()

    def forward(self, x: t.Tensor):
        return self.generator(x)

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
            # ipdb.set_trace()
            pred_temp_label = self.temporal_discriminator(t.cat((x, fake_y), dim=1))
            real_frame_label = t.ones(batch_size * y_seq_len).to(self.device)
            real_temp_label = t.ones(batch_size).to(self.device)

            generator_loss = (
                self.adversarial_loss(pred_temp_label, real_temp_label) *0.5
                + self.adversarial_loss(pred_frame_label, real_frame_label) *0.5
                # + self.l1_loss(fake_y, y).mean() * 0.05
            ) 

            if batch_idx % 50 == 0:
                visualize_predictions(
                    x, y, fake_y, self.current_epoch, self.params.save_path
                )

            train_mse = F.mse_loss(fake_y, y)
            self.log("train_mse", train_mse, prog_bar=True)
            return {
                "loss": generator_loss,
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
            # labels = t.cat((real_label, fake_label)).squeeze()
            # frames = t.cat((y_frames, fake_frames))
            frame_disc_loss = (
                self.adversarial_loss(self.frame_discriminator(fake_frames), fake_label)
                + self.adversarial_loss(self.frame_discriminator(y_frames), real_label)
            ) * 0.5
            self.log("fd_loss", frame_disc_loss, prog_bar=True)

            return {
                "loss": frame_disc_loss,
            }

        # train temporal discriminator
        if optimizer_idx == 2:
            real_label = t.ones(batch_size, 1, device=self.device)
            fake_label = t.zeros(batch_size, 1, device=self.device)
            labels = t.cat((real_label, fake_label)).squeeze()
            fake_sequence = t.cat((x, self.fake_y_detached), dim=1)
            real_sequence = t.cat((x, y), dim=1)
            # sequences = t.cat((real_sequence, fake_sequence))
            # pred_labels = self.temporal_discriminator(sequences)
            temp_disc_loss = (
                self.adversarial_loss(
                    self.temporal_discriminator(fake_sequence), fake_label
                )
                + self.adversarial_loss(
                    self.temporal_discriminator(real_sequence), real_label
                )
            ) * 0.5

            self.log("td_loss", temp_disc_loss, prog_bar=True)
            return {
                "loss": temp_disc_loss,
            }

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x[0]["train_mse"] for x in outputs]).mean()
        fd_loss = t.stack([x[1]["loss"] for x in outputs]).mean()
        td_loss = t.stack([x[2]["loss"] for x in outputs]).mean()
        self.log("train_mse", avg_loss, prog_bar=True)
        # self.log("train_fd", fd_loss, prog_bar=True)
        # self.log("train_td", td_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

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
