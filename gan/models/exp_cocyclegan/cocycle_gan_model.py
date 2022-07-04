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
        return F.binary_cross_entropy(y_hat.flatten(), y.flatten()).mean()

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
            self.generator(self.__embed(y, future=self.x_future)),
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
        sum_mse_loss = t.stack([x["val_mse_loss"] for x in outputs]).sum()
        self.val_mse.reset()

        self.log("val_sum_mse_loss", sum_mse_loss)
        self.log("val_loss", avg_mse_loss, prog_bar=True)

        # self.log("val_loss", avg_loss, prog_bar=True)
        # self.log("val_gen_loss", avg_loss, prog_bar=True)
        # self.log("val_disc_loss", avg_disc_loss, prog_bar=True)

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
            fake_x = self.generator(self.__embed(y, future=not future))
            true_label = t.ones(batch_size, 1, device=self.device)
            if future:
                sequence = t.cat((x, fake_y), 1)
            else:
                sequence = t.cat((fake_x, y), 1)

            generator_loss_y = self.__adversarial_loss(
                self.discriminator(sequence),
                true_label,
            )
            return generator_loss_y
        else:
            return 0

    def __discriminator_loss(self, x: t.Tensor, y: t.Tensor):

        return self.__all_way_discriminator_loss(x, y) / 2

        forward_discriminator_loss = self.__one_way_discriminator_loss(
            x, y, future=self.x_future
        )
        backward_discriminator_loss = self.__one_way_discriminator_loss(
            x, y, future=self.y_future
        )
        cyclic_discriminator_loss = self.__cyclic_way_discriminator_loss(x, y)

        return (
            forward_discriminator_loss
            + backward_discriminator_loss
            + cyclic_discriminator_loss
        ) / 3

    def __one_way_discriminator_loss(self, x: t.Tensor, y: t.Tensor, future: bool):
        batch_size = x.shape[0]
        true_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)

        fake_y = self.generator(self.__embed(x if future else y, future=future))
        if future:
            sequence = t.cat((x, fake_y), dim=1)
        else:
            sequence = t.cat((fake_y, x), dim=1)

        pred_x_true_labels = self.discriminator(t.cat((x, y), dim=1))
        pred_y_fake_labels_to_future = self.discriminator(sequence)
        return (
            self.__adversarial_loss(pred_x_true_labels, true_label)
            + self.__adversarial_loss(pred_y_fake_labels_to_future, fake_label)
        ) / 2

    def __cyclic_way_discriminator_loss(self, x: t.Tensor, y: t.Tensor):
        batch_size = x.shape[0]
        true_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)

        fake_y = self.generator(self.__embed(x, future=self.y_future))
        fake_x = self.generator(self.__embed(fake_y, future=self.x_future))
        sequence = t.cat((fake_x, fake_y), dim=1)
        pred_x_true_labels = self.discriminator(t.cat((x, y), dim=1))
        pred_y_fake_labels_to_future = self.discriminator(sequence)
        return (
            self.__adversarial_loss(pred_x_true_labels, true_label)
            + self.__adversarial_loss(pred_y_fake_labels_to_future, fake_label)
        ) / 2

    def __cyclic_way_generator_loss(self, x: t.Tensor, y: t.Tensor):
        batch_size = x.shape[0]
        true_label = t.ones(batch_size, 1, device=self.device)

        fake_y = self.generator(self.__embed(x, future=self.y_future))
        fake_x = self.generator(self.__embed(fake_y, future=self.x_future))
        sequence = t.cat((fake_x, fake_y), dim=1)
        return self.__adversarial_loss(self.discriminator(sequence), true_label)

    def __all_way_discriminator_loss(self, x: t.Tensor, y: t.Tensor):

        b_size = x.shape[0]

        fake_y = self.generator(self.__embed(x, future=True))
        fake_x = self.generator(self.__embed(fake_y, future=False))
        pred_x = self.generator(self.__embed(y, future=False))

        first_b = t.cat((x, fake_y), 1)
        # second_b = t.cat((fake_x, y), 1)
        third_b = t.cat((pred_x, y), 1)
        # fourth_b = t.cat((fake_x, fake_y), 1)

        all_fakes = t.cat((first_b, third_b), 0)

        # randomly select b_size from all_fakes
        all_fakes = all_fakes[t.randperm(all_fakes.shape[0])[:b_size]]

        batch_size = all_fakes.shape[0]
        true_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)

        true_sequence = t.cat((x, y), 1)

        return self.__adversarial_loss(
            self.discriminator(all_fakes),
            fake_label,
        ) + self.__adversarial_loss(self.discriminator(true_sequence), true_label)

    def __three_way_generator_loss(
        self,
        x: t.Tensor,
        y: t.Tensor,
    ):
        fake_y = self.generator(self.__embed(x, future=True))
        fake_x = self.generator(self.__embed(fake_y, future=False))
        pred_x = self.generator(self.__embed(y, future=False))

        first_b = t.cat((x, fake_y), 1)
        # second_b = t.cat((fake_x, y), 1)
        third_b = t.cat((pred_x, y), 1)
        # fourth_b = t.cat((fake_x, fake_y), 1)

        all_fakes = t.cat((first_b, third_b), 0)

        batch_size = all_fakes.shape[0]
        true_label = t.ones(batch_size, 1, device=self.device)
        generator_loss_y = self.__adversarial_loss(
            self.discriminator(all_fakes),
            true_label,
        )
        return generator_loss_y

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
        fake_x = self.generator(self.__embed(pred_y, future=self.x_future))
        pred_x = self.generator(self.__embed(y, future=self.x_future))
        cycle_y = self.generator(self.__embed(pred_x, future=self.y_future))

        pred_y_l1 = F.smooth_l1_loss(pred_y, y)
        pred_x_l1 = F.smooth_l1_loss(pred_x, x)
        pred_cycle_l1 = F.smooth_l1_loss(fake_x, x)
        pred_y_cycle_l1 = F.smooth_l1_loss(cycle_y, y)

        mse_sum_pred_y_loss = F.mse_loss(pred_y, y, reduction="sum")
        adversarial_loss = self.__three_way_generator_loss(x, y)

        # pred_y_frame_one_loss = F.l1_loss(pred_y[:, 0, :], y[:, 0, :])
        # pred_x_frame_one_loss = F.l1_loss(pred_x[:, 0, :], x[:, 0, :])

        # pred_y_frame_last_loss = F.l1_loss(pred_y[:, -1, :], y[:, -1, :])
        # pred_x_frame_last_loss = F.l1_loss(pred_x[:, -1, :], x[:, -1, :])

        # pred_y_frame_middle_loss = F.l1_loss(pred_y[:, 5, :], y[:, 5, :])
        # pred_x_frame_middle_loss = F.l1_loss(pred_x[:, 5, :], x[:, 5, :])

        # frame_middle_loss = pred_y_frame_middle_loss + pred_x_frame_middle_loss

        # self.log("CoCycleLoss/pred_y_frame_one_loss", pred_y_frame_one_loss)
        # self.log("CoCycleLoss/pred_x_frame_one_loss", pred_x_frame_one_loss)
        # self.log("CoCycleLoss/pred_y_frame_last_loss", pred_y_frame_last_loss)
        # self.log("CoCycleLoss/pred_x_frame_last_loss", pred_x_frame_last_loss)
        self.log("CoCycleLoss/pred_y_l1", pred_y_l1)
        self.log("CoCycleLoss/adversarial_loss", adversarial_loss)
        self.log("CoCycleLoss/pred_y", pred_y_l1)
        self.log("CoCycleLoss/pred_x", pred_x_l1)
        self.log("CoCycleLoss/pred_cycle", pred_cycle_l1)
        self.log("CoCycleLoss/pred_y_cycle", pred_y_cycle_l1)
        self.log("CoCycleLoss/mse_sum_pred_y", mse_sum_pred_y_loss / 10)

        # loss = (
        #     +self.params.hparams["l1_cyclic_loss_weight"]
        #     * (pred_cycle_l1 + pred_y_cycle_l1)
        #     + self.params.hparams["l1_pred_loss_weight"] * pred_y_l1
        #     + self.params.hparams["l1_past_pred_loss_weight"] * pred_x_l1
        # )

        # max = loss * 1e-3
        # adversarial_loss = max * adversarial_loss
        # adversarial_loss = adversarial_loss.max(max)

        # return loss + adversarial_loss

        l2_lambda = 0.0001
        l2_norm = sum(p.pow(2).sum() for p in self.generator.parameters())
        self.log("CoCycleLoss/l1_norm", l2_lambda * l2_norm, prog_bar=True)
        # self.log("OneWayLoss/l1", F.smooth_l1_loss(pred_y, y), prog_bar=True)
        # self.log(
        #     "OneWayLoss/pred_y", (F.mse_loss(pred_y, y) + 10 * pred_y_l1), prog_bar=True
        # )
        return (
            self.params.hparams["adversarial_loss_weight"] * adversarial_loss
            + self.params.hparams["l1_cyclic_loss_weight"]
            * (pred_cycle_l1 + pred_y_cycle_l1)
            + self.params.hparams["l1_pred_loss_weight"]
            * (F.mse_loss(pred_y, y) )
            + self.params.hparams["l1_past_pred_loss_weight"]
            * (F.mse_loss(pred_x, x) )
            + l2_norm * l2_lambda
            # + self.params.hparams["l1_pred_frame_one_loss_weight"]
            # * (pred_y_frame_one_loss + pred_x_frame_one_loss)
            # + self.params.hparams["l1_pred_frame_last_loss_weight"]
            # * (pred_y_frame_last_loss + pred_x_frame_last_loss)
            # + self.params.hparams["l1_pred_frame_middle_loss_weight"] * frame_middle_loss
        )

    def training_step(
        self,
        batch: tuple[t.Tensor, t.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        x, y = batch

        if batch_idx % self.params.save_interval == 0:
            # save the model on a thread
            self.__save_model()

        # train generator
        if optimizer_idx == 0:

            pred = self.generator(self.__embed(x, future=self.y_future))
            mse_loss = F.mse_loss(pred, y)
            self.log("mse_loss", mse_loss, prog_bar=True)
            loss = self.__generator_loss(x, y, batch_idx, flag="train")
            self.log("g_loss", loss, prog_bar=True)
            return {"loss": loss}

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
