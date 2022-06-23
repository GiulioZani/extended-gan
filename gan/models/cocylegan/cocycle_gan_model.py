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

from gan.models.cocylegan.visualize import _visualize_predictions

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

        self.__cyclic_function = {
            "l1": F.l1_loss,
            "bce": self.__norm_bce,
        }[self.params.cycle_similarity_function]
        self.x_future = False
        self.y_future = True
        self.to_visualize = [
            (t.tensor(0), t.tensor(0), t.tensor(0)),
            (t.tensor(0), t.tensor(0), t.tensor(0)),
        ]

    def __norm_bce(self, x, y):
        norm_x = (x - x.min()) / (x.max() - x.min())
        norm_y = (y - y.min()) / (y.max() - y.min())
        print("hereeee")
        uba = F.binary_cross_entropy(norm_x, norm_y)
        print(f"{uba=}")
        return uba

    def forward(self, x: t.Tensor):
        return self.generator(x)

    def __adversarial_loss(self, y_hat: t.Tensor, y: t.Tensor):
        # squeeze if shapes are (batch_size, 1)
        if y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze()
        if y.shape[-1] == 1:
            y = y.squeeze()
        return F.binary_cross_entropy(y_hat, y).mean()

    def __cyclic_loss(self, real_x: t.Tensor, fake_x: t.Tensor):
        return (
            0
            if self.params.cycle_similarity_weight == 0
            else (
                self.params.cycle_similarity_weight
                * self.__cyclic_function(real_x, fake_x).mean()
            )
        )

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

    def __unembed(self, x: t.Tensor) -> t.Tensor:
        return x[:, 1:]

    def __visualize_predictions(self, x, y, flag):
        future = self.y_future
        fake_y = self.generator(self.__embed(x, future=future))
        fake_x = self.generator(self.__embed(fake_y, future=not future))

        _visualize_predictions(
            x,
            y,
            fake_y,
            fake_x,
            self.current_epoch,
            self.params.save_path,
            flag,
        )

        # nrows = min(nrows, x.shape[0])
        return
        nrows = 2
        _, axes = plt.subplots(nrows=nrows, ncols=3)
        for row in range(nrows):
            x, fake_y, fake_x = self.to_visualize[row]
            assert (x.shape == fake_x.shape == fake_y.shape) and x.shape[
                1
            ] == 3, f"Wrong shapes: {x.shape=} {fake_y.shape=} {fake_x.shape=}"

            x_row = (x[row] - x[row].min()) / (x[row].max() - x[row].min())
            fake_y_row = fake_y[row]
            fake_x_row = fake_x[row]
            axes[row][0].imshow(x_row.permute(1, 2, 0).cpu().numpy())
            axes[row][0].set_title("horse" if row == 0 else "zebra")
            axes[row][1].imshow((fake_y_row.permute(1, 2, 0).cpu().numpy() + 1) / 2)
            axes[row][1].set_title("zebra" if row == 0 else "horse")
            axes[row][2].imshow((fake_x_row.permute(1, 2, 0).cpu().numpy() + 1) / 2)
            axes[row][2].set_title("horse" if row == 0 else "zebra")
        plt.savefig(
            os.path.join(
                self.params.save_path,
                f"{flag}_prediction.png",
            )
        )
        plt.close()

    def validation_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        generator_loss = self.__generator_loss(x, y, batch_idx, flag="val")
        discriminator_loss = self.__discriminator_loss(x, y)
        pred = self.generator(self.__embed(x, future=self.y_future))
        mse_loss = F.mse_loss(pred, y)
        self.val_mse.update(pred.detach(), y.detach())

        if batch_idx == 0:
            self.__visualize_predictions(x, y, "val")

        return {
            "val_loss": generator_loss,
            "val_discriminator_loss": discriminator_loss,
            "val_mse_loss": mse_loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        avg_disc_loss = t.stack([x["val_discriminator_loss"] for x in outputs]).mean()
        avg_mse_loss = self.val_mse.compute()
        self.val_mse.reset()

        self.log("r_val_loss", avg_mse_loss, prog_bar=True)
        

        # self.log("val_loss", avg_loss, prog_bar=True)
        # self.log("val_gen_loss", avg_loss, prog_bar=True)
        # self.log("val_disc_loss", avg_disc_loss, prog_bar=True)
        self.log(
            "val_mse_loss",
            t.stack([x["val_mse_loss"] for x in outputs]).mean(),
            prog_bar=True,
        )
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            t.save(
                self.state_dict(),
                os.path.join(self.params.save_path, "best_model.pt"),
            )
            print("Saved model")

        return {"val_loss": avg_loss}

    def test_step(self, batch: tuple[t.Tensor, t.Tensor], batch_idx: int):
        x, y = batch
        generator_loss = self.__generator_loss(x, y, batch_idx, flag="test")
        discriminator_loss = self.__discriminator_loss(x, y)
        pred = self.generator(self.__embed(x, future=self.y_future))
        mse_loss = F.mse_loss(pred, y)

        self.val_mse.update(pred.detach(), y.detach())

        if batch_idx == 0:
            self.__visualize_predictions(x, y, "test")
        return {
            "val_loss": generator_loss,
            "val_discriminator_loss": discriminator_loss,
            "val_mse_loss": mse_loss,
        }

    def test_epoch_end(self, outputs):
        avg_loss = t.stack([x["val_loss"] for x in outputs]).mean()
        avg_disc_loss = t.stack([x["val_discriminator_loss"] for x in outputs]).mean()
        self.log("val_mse_loss", t.stack([x["val_mse_loss"] for x in outputs]).mean())
        # self.log("val_loss", avg_loss, prog_bar=True)
        # self.log("val_gen_loss", avg_loss, prog_bar=True)
        # self.log("val_disc_loss", avg_disc_loss, prog_bar=True)
        self.log("r_val_mse_loss", self.val_mse.compute())
        self.val_mse.reset()
        return {"val_loss": avg_loss}

    def __one_way_generator_loss(
        self,
        x: t.Tensor,
        y: t.Tensor,
        future=False,
        visualize=False,
    ):
        if (future and self.params.backward_generator_loss) or (
            not future and self.params.forward_generator_loss
        ):
            batch_size = x.shape[0]
            fake_y = self.generator(self.__embed(x, future=future))
            fake_x = self.generator(self.__embed(fake_y, future=not future))
            true_label = t.ones(batch_size, 1, device=self.device)
            # generator_loss_x = self.__adversarial_loss(
            #    self.discriminator(self.__embed(fake_x, future=future)),
            #    true_label,
            # )
            if future:
                sequence = t.cat((x, fake_y), 1)
            else:
                sequence = t.cat((y, fake_x), 1)

            generator_loss_y = self.__adversarial_loss(
                self.discriminator(sequence),
                true_label,
            )

            # if visualize:
            #     self.to_visualize[future] = (
            #         x.cpu().detach(),
            #         fake_y.cpu().detach(),
            #         fake_x.cpu().detach(),
            #     )
            return generator_loss_y + self.__cyclic_loss(x, (fake_x))
        else:
            return 0

    def __discriminator_loss(self, x: t.Tensor, y: t.Tensor):
        forward_discriminator_loss = self.__one_way_discriminator_loss(
            x, y, future=self.x_future
        )
        backward_discriminator_loss = self.__one_way_discriminator_loss(
            y, y, future=self.y_future
        )
        return (forward_discriminator_loss + backward_discriminator_loss) / 2

    def __one_way_discriminator_loss(self, x: t.Tensor, y: t.Tensor, future: bool):
        batch_size = x.shape[0]
        true_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)

        fake_y = self.generator(self.__embed(x, future=future))
        if future:
            sequence = t.cat((x, fake_y), dim=1)
        else:
            sequence = t.cat((fake_y, x), dim=1)

        pred_x_true_labels = self.discriminator(t.cat((x, y), dim=1))
        pred_y_fake_labels_to_future = self.discriminator(sequence)
        return (
            self.__adversarial_loss(pred_x_true_labels, true_label)
            # + self.__adversarial_loss(pred_x_fake_labels, fake_label)
            + self.__adversarial_loss(pred_y_fake_labels_to_future, fake_label)
        ) / 2

    def __generator_loss(self, x: t.Tensor, y: t.Tensor, batch_idx: int, flag: str):
        visualize = (batch_idx % 10 == 0) if flag == "train" else (batch_idx == 0)

        forward_generator_loss = self.__one_way_generator_loss(
            x, y, future=self.x_future, visualize=visualize
        )
        backward_generator_loss = self.__one_way_generator_loss(
            y, y, future=self.y_future, visualize=visualize
        )
        if visualize:
            self.__visualize_predictions(x, y, flag)
        return forward_generator_loss + backward_generator_loss + F.mse_loss(x, y)

    def training_step(
        self,
        batch: tuple[t.Tensor, t.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        x, y = batch

        if batch_idx % self.params.save_interval == 0:
            # save the model
            t.save(
                self.state_dict(),
                os.path.join(self.params.save_path, "model.pt"),
            )

        # train generator
        if optimizer_idx == 0:

            pred = self.generator(self.__embed(x, future=self.y_future))
            mse_loss = F.mse_loss(pred, y)
            self.log("train_mse_loss", mse_loss, prog_bar=True)

            return {
                "loss": self.__generator_loss(x, y, batch_idx, flag="train"),
            }
        # train temporal discriminator
        if optimizer_idx == 1:
            loss = self.__discriminator_loss(x, y)
            self.log("TD_Loss", loss, True)
            return {
                "loss": loss,
            }

    def training_epoch_end(self, outputs):
        avg_gen_loss = t.stack([x[0]["loss"] for x in outputs]).mean()
        avg_disc_loss = t.stack([x[1]["loss"] for x in outputs]).mean()
        self.log("gen_loss", avg_gen_loss, prog_bar=True)
        self.log("disc_loss", avg_disc_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        discriminator_optimizer = t.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        # initializes reduce_lr_on_plateau scheduler
        generator_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer,
            patience=self.params.reduce_lr_on_plateau_patience,
            verbose=True,
            factor=self.params.reduce_lr_on_plateau_factor,
        )
        discriminator_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
            discriminator_optimizer,
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
            {
                "optimizer": discriminator_optimizer,
                "lr_scheduler": {
                    "scheduler": discriminator_scheduler,
                    "monitor": "val_loss",
                },
            },
        ]
