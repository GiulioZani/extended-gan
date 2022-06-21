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
from .sam import SAM, disable_running_stats, enable_running_stats


class BaseGanLightning(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()

        self.generator = nn.Sequential()
        self.frame_discriminator = nn.Sequential()
        self.temporal_discriminator = nn.Sequential()
        self.fake_y_detached = t.tensor(0.0)
        self.automatic_optimization = False

    def adversarial_loss(self, y_hat: t.Tensor, y: t.Tensor):
        return F.binary_cross_entropy(y_hat.flatten(), y.flatten()).mean()

    def compute_frame_discriminator_loss():
        pass

    def generator_loss(self, batch):
        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape
        fake_y = self(x)

        fake_data_frames = fake_y.reshape(
            batch_size * y_seq_len, channels, height, width
        )
        pred_frame_label = self.frame_discriminator(fake_data_frames)
        pred_temp_label = self.temporal_discriminator(t.cat((x, fake_y), dim=1))
        real_frame_label = t.ones(batch_size * y_seq_len).to(self.device)
        real_temp_label = t.ones(batch_size).to(self.device)

        generator_loss = (
            self.adversarial_loss(pred_temp_label, real_temp_label)
            + self.adversarial_loss(pred_frame_label, real_frame_label)
        ) * 0.5

        return generator_loss

    def frame_discriminator_loss(self, batch):
        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape

        y_frames = y.reshape(batch_size * y_seq_len, channels, height, width)
        fake_frames = self.fake_y_detached.reshape(
            batch_size * y_seq_len, channels, height, width
        )
        y_batch_size = batch_size * y_seq_len
        real_label = t.ones(y_batch_size, 1, device=self.device)
        fake_label = t.zeros(y_batch_size, 1, device=self.device)

        frame_disc_loss = (
            self.adversarial_loss(self.frame_discriminator(fake_frames), fake_label)
            + self.adversarial_loss(self.frame_discriminator(y_frames), real_label)
        ) * 0.5

        return frame_disc_loss

    def frame_discriminator_optimizer(self, batch):
        optimizer = self.optimizers()[1]

        loss = self.frame_discriminator_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      
        return loss

    def temporal_discriminator_loss(self, batch):

        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape

        real_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)

        fake_sequence = t.cat((x, self.fake_y_detached), dim=1)
        real_sequence = t.cat((x, y), dim=1)

        temp_disc_loss = (
            self.adversarial_loss(
                self.temporal_discriminator(fake_sequence), fake_label
            )
            + self.adversarial_loss(
                self.temporal_discriminator(real_sequence), real_label
            )
        ) * 0.5
        return temp_disc_loss

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        if batch_idx == 0:
            visualize_predictions(
                x, y, self(x), path=self.params.save_path + "/validation"
            )
        y = y.cpu()
        pred_y = self(x).cpu()
        loss = F.mse_loss(pred_y, y)
        # self.log("val_mse", loss, prog_bar=True)
        return {"val_mse": loss, "val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        t.save(
            self.state_dict(),
            os.path.join(self.params.save_path, "model_val.pt"),
        )
        return {"val_loss": avg_loss}


    def temporal_discriminator_optimizer(self, batch):
        optimizer = self.optimizers()[2]

        # enable_running_stats(self.temporal_discriminator)
        # loss_1 = self.temporal_discriminator_loss(batch)
        # self.manual_backward(loss_1)
        # optimizer.first_step(zero_grad=True)

        # disable_running_stats(self.temporal_discriminator)
        # loss_2 = self.temporal_discriminator_loss(batch)
        # self.manual_backward(loss_2)
        # optimizer.second_step(zero_grad=True)
        loss = self.temporal_discriminator_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss

    def gen_sam_optimizer(self, batch):
        optimizer = self.optimizers()[0]

        enable_running_stats(self.generator)
        loss_1 = self.generator_loss(batch)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        disable_running_stats(self.generator)
        loss_2 = self.generator_loss(batch)
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)


        enable_running_stats(self.generator)
        loss_1 = F.mse_loss(self(batch[0]), batch[1])
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        disable_running_stats(self.generator)
        loss_2 = F.mse_loss(self(batch[0]), batch[1])
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)

        

        enable_running_stats(self.generator)
        loss_1 = self.generator_loss(batch)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        disable_running_stats(self.generator)
        loss_2 = self.generator_loss(batch)
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)
        return loss_2

    def forward(self, x: t.Tensor):
        return self.generator(x)

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape
        # train generator
        optimizer_idx = 0
        if optimizer_idx == 0:

            generator_loss = self.gen_sam_optimizer(batch)
            fake_y = self(x)
            # i save it as property to avoid recomputing it for the discriminators
            self.fake_y_detached = fake_y.detach()  # used later by discriminators

            if batch_idx % 50 == 0:
                
                visualize_predictions(
                    x, y, fake_y, self.current_epoch, self.params.save_path
                )

            train_mse = F.mse_loss(fake_y, y)
            self.log("train_mse", train_mse, prog_bar=True)
            # return {
            #     "loss": generator_loss,
            # }

        # train frame discriminator
        optimizer_idx = 1
        if optimizer_idx == 1:

            frame_disc_loss = self.frame_discriminator_optimizer(batch)
            self.log("fd_loss", frame_disc_loss, prog_bar=True)

            # return {
            #     "loss": frame_disc_loss,
            # }

        # train temporal discriminator
        optimizer_idx = 2
        if optimizer_idx == 2:
            temp_disc_loss = self.temporal_discriminator_optimizer(batch)
            self.log("td_loss", temp_disc_loss, prog_bar=True)
            # return {
            #     "loss": temp_disc_loss,
            # }
        
    def training_epoch_end(self, outputs):
        pass
        # avg_loss = t.stack([x[0]["train_mse"] for x in outputs]).mean()
        # fd_loss = t.stack([x[1]["loss"] for x in outputs]).mean()
        # td_loss = t.stack([x[2]["loss"] for x in outputs]).mean()
        # self.log("train_mse", avg_loss, prog_bar=True)
        # self.log("train_fd", fd_loss, prog_bar=True)
        # self.log("train_td", td_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = SAM(
            self.generator.parameters(), t.optim.SGD, lr=0.001, momentum=0.9
        )

        # generator_optimizer = t.optim.Adam(
        #     self.generator.parameters(), lr=lr, betas=(b1, b2)
        # )
        frame_discriminator_optimizer = t.optim.Adam(
            self.frame_discriminator.parameters(), lr=lr, betas=(b1, b2))

        temporal_discriminator_optimizer = t.optim.Adam(
            self.temporal_discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (
            [
                generator_optimizer,
                frame_discriminator_optimizer,
                temporal_discriminator_optimizer,
            ],
            [],
        )
