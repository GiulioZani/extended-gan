from turtle import forward
from pytorch_lightning import LightningModule
import torch as t
import torch.nn.functional as F
from argparse import Namespace
from torchmetrics import Accuracy
import torch.nn as nn
import torchmetrics
from .visualize import visualize_predictions
import matplotlib.pyplot as plt
import ipdb
import os
from .sam import SAM, disable_running_stats, enable_running_stats


class BaseMetricTracker:
    def __init__(self, metric_names):
        self.metric_names = metric_names
        # self.metrics = {
        #     "MSE": torchmetrics.MeanSquaredError(),
        #     "MAE": torchmetrics.MeanAbsoluteError(),
        # }
        self.frame_metics = [
            {
                f"MSE_{i}": torchmetrics.MeanSquaredError(),
                f"MAE_{i}": torchmetrics.MeanAbsoluteError(),
            }
            for i in range(10)
        ]
        self.frame_metics.append(
            {
                f"MSE": torchmetrics.MeanSquaredError(),
                f"MAE": torchmetrics.MeanAbsoluteError(),
            }
        )

    def update(self, y_hat, y):
        # for metric_name in self.metrics.keys():
        #     self.metrics[metric_name].update(y_hat, y)

        for i in range(10):
            self.frame_metics[i][f"MSE_{i}"].update(
                y_hat[:, i, ...].detach().cpu(), y[:, i, ...].detach().cpu()
            )
            self.frame_metics[i][f"MAE_{i}"].update(
                y_hat[:, i, ...].detach().cpu(), y[:, i, ...].detach().cpu()
            )
        self.frame_metics[10]["MSE"].update(y_hat.detach().cpu(), y.detach().cpu())
        self.frame_metics[10]["MAE"].update(y_hat.detach().cpu(), y.detach().cpu())

    def get_metrics(self):

        # metrics = {
        #     metric_name: self.metrics[metric_name].compute()
        #     for metric_name in self.metric_names.keys()
        # }
        metrics = {}
        for i in range(10):
            metrics[f"MSE_{i}"] = self.frame_metics[i][f"MSE_{i}"].compute()
            metrics[f"MAE_{i}"] = self.frame_metics[i][f"MAE_{i}"].compute()

        metrics["MSE"] = self.frame_metics[10]["MSE"].compute()
        metrics["MAE"] = self.frame_metics[10]["MAE"].compute()

        self.print_metrics_latex(metrics)
        return metrics


    # tabular latex format
    def print_latex_matrix(self, column_labels, values):

        print("begin{tabular}{|l|", end="")
        for i in range(len(column_labels)):
            print("c|", end="")
        print("}")
        print("hline", end="")
        for i in range(len(column_labels)):
            print("&", end="")
            print(column_labels[i], end="")
        print("\\\\")
        print("hline", end="")
        for i in range(len(column_labels)):
            print("&", end="")
            print("{:.2f}".format(values[i]), end="")
        print("\\\\")
        print("hline", end="")
        print("end{tabular}")



    def print_metrics_latex(self,metrics):
        # metrics = self.get_metrics()
        
        # each key in metrics is a column
        lables = metrics.keys()
        values = metrics.values()
        self.print_latex_matrix(lables, values)


        


class BaseGanLightning(LightningModule):
    def __init__(self, params: Namespace):
        super().__init__()
        self.params = params
        self.save_hyperparameters()

        self.generator = nn.Sequential()
        self.frame_discriminator = nn.Sequential()
        self.temporal_discriminator = nn.Sequential()
        self.metrics = BaseMetricTracker("test")
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
        real_frame_label = t.ones(batch_size * y_seq_len).to(self.device)
        real_temp_label = t.ones(batch_size).to(self.device)

        generator_loss = (
            self.adversarial_loss(pred_frame_label, real_frame_label)
            + self.adversarial_loss(
                self.temporal_discriminator(t.cat([x, fake_y], 1)), real_temp_label
            )
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

        # enable_running_stats(self.frame_discriminator)
        # zero grad
        optimizer.zero_grad()
        # compute loss
        loss = self.frame_discriminator_loss(batch)
        # compute gradients
        loss.backward()
        # update weights
        optimizer.step()

        return loss

    def temporal_discriminator_loss(self, discriminator, batch):

        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape

        real_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)

        fake_sequence = t.cat((x, self.fake_y_detached), dim=1)
        real_sequence = t.cat((x, y), dim=1)

        temp_disc_loss = (
            self.adversarial_loss(discriminator(fake_sequence), fake_label)
            + self.adversarial_loss(discriminator(real_sequence), real_label)
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

    def temporal_discriminator_optimizer(
        self, optimizer_id, temporal_discriminator, batch
    ):
        optimizer = self.optimizers()[optimizer_id]

        # if optimizer_id == 2:
        #     optimizer.zero_grad()
        #     loss = self.temporal_discriminator_loss(temporal_discriminator, batch)
        #     loss.backward()
        #     optimizer.step()
        #     return loss

        enable_running_stats(temporal_discriminator)
        loss_1 = self.temporal_discriminator_loss(temporal_discriminator, batch)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        disable_running_stats(temporal_discriminator)
        loss_2 = self.temporal_discriminator_loss(temporal_discriminator, batch)
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)
        return loss_1

    def gen_sam_optimizer(self, batch):
        optimizer = self.optimizers()[0]
        x, y = batch

        enable_running_stats(self.generator)
        loss_1 = F.mse_loss(self.generator(x), y)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)

        disable_running_stats(self.generator)
        loss_2 = F.mse_loss(self.generator(x), y)
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

        return loss_1

    def forward(self, x: t.Tensor):
        return self.generator(x)

    def training_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch

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
            self.log("train_loss", generator_loss, prog_bar=True)
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
            temp_disc_loss = self.temporal_discriminator_optimizer(
                2, self.temporal_discriminator, batch
            )
            self.log("td_loss", temp_disc_loss, prog_bar=True)
            # return {
            #     "loss": temp_disc_loss,
            # }

        # optimizer_idx = 3
        # if optimizer_idx == 3:
        #     td2_loss = self.temporal_discriminator_optimizer(
        #         3, self.second_temporal_discriminator, batch
        #     )
        #     self.log("td2_loss", td2_loss, prog_bar=True)

        if batch_idx % self.params.save_interval == 0:
            t.save(
                self.state_dict(),
                os.path.join(self.params.save_path, "model.pt"),
            )

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        self.metrics.update(self(x), y)

    def test_epoch_end(self, outputs):
        return self.metrics.get_metrics()

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
            self.generator.parameters(), t.optim.SGD, lr=lr, momentum=0.9
        )

        # generator_optimizer = t.optim.Adam(
        #     self.generator.parameters(), lr=lr, betas=(b1, b2)
        # )
        frame_discriminator_optimizer = t.optim.Adam(
            self.frame_discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        #  SAM(
        #     self.frame_discriminator.parameters(), t.optim.SGD, lr=lr, momentum=0.9
        # )
        temporal_discriminator_optimizer = SAM(
            self.temporal_discriminator.parameters(),
            t.optim.SGD,
            lr=lr,
            momentum=0.9,
        )
        second_temporal_discriminator_optimizer = SAM(
            self.second_temporal_discriminator.parameters(),
            t.optim.SGD,
            lr=lr,
            momentum=0.9,
        )
        return (
            [
                generator_optimizer,
                frame_discriminator_optimizer,
                temporal_discriminator_optimizer,
                # second_temporal_discriminator_optimizer,
            ],
            [],
        )