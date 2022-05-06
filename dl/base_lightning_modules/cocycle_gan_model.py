import torch as t
import torch.nn as nn
import torch.nn.functional as F

# from .base_model import BaseModel
import ipdb
from argparse import Namespace

from .base_model import BaseRegressionModel


class CoCycleGAN(BaseRegressionModel):
    def __init__(self, params: Namespace):
        super().__init__(params)
        self.params = params
        self.temporal_discriminator = nn.Sequential()
        self.temporal_embedding = nn.Embedding(2, self.params.imsize ** 2)
        # self.fake_y_detached = t.tensor(0.0)
        # self.fake_x_detached = t.tensor(0.0)
        self.real_sequence_detached = t.tensor(0.0)
        self.fake_sequence_detached = t.tensor(0.0)
        self.real_sequence_reversed_detached = t.tensor(0.0)
        self.fake_sequence_reversed_detached = t.tensor(0.0)

    def adversarial_loss(self, y_hat: t.Tensor, y: t.Tensor):
        # squeeze if shapes are (batch_size, 1)
        if y_hat.shape[-1] == 1:
            y_hat = y_hat.squeeze()
        if y.shape[-1] == 1:
            y = y.squeeze()
        return F.binary_cross_entropy(y_hat, y)

    def embed(self, x: t.Tensor, future: bool = False):
        label = (t.tensor(1) if future else t.tensor(0)).to(self.device)
        embedding = (
            self.temporal_embedding(label)
            .view(self.params.imsize, self.params.imsize)
            .repeat((x.shape[0], x.shape[1], 1, 1))
            .unsqueeze(2)
        ).to(self.device)
        return t.cat((x, embedding), dim=2)

    def one_way_generator_loss(
        self, fake_sequence: t.Tensor, future: bool,
    ):
        pred_temp_label = self.temporal_discriminator(
            fake_sequence, future=future
        )
        real_temp_label = t.ones(fake_sequence.shape[0]).to(self.device)
        generator_loss = self.adversarial_loss(
            pred_temp_label, real_temp_label
        )
        return generator_loss

    def one_way_temporal_discriminator_loss(
        self, real_sequence: t.Tensor, fake_sequence: t.Tensor, future: bool,
    ):
        batch_size = real_sequence.shape[0]
        real_label = t.ones(batch_size, 1, device=self.device)
        fake_label = t.zeros(batch_size, 1, device=self.device)
        labels = t.cat((real_label, fake_label)).squeeze()
        sequences = t.cat((real_sequence, fake_sequence))
        pred_labels = self.temporal_discriminator(sequences)
        temp_disc_loss = self.adversarial_loss(pred_labels, labels)
        return temp_disc_loss

    def training_step(
        self,
        batch: tuple[t.Tensor, t.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ):
        x, y = batch
        batch_size, x_seq_len, channels, height, width = x.shape
        batch_size, y_seq_len, channels, height, width = y.shape
        # train generator
        if optimizer_idx == 0:
            fake_y = self(self.embed(x, future=True))
            # self.fake_y_detached = fake_y.detach()  # used later by discriminators
            fake_x = self.generator(
                self.embed(t.flip(fake_y, dims=(1,)), future=False)
            )
            # self.fake_x_detached = fake_x.detach()  # used later by discriminators
            # i save it as property to avoid recomputing it for the discriminators
            real_sequence = self.embed(t.cat((x, y), dim=1))
            fake_sequence = self.embed(t.cat((x, fake_y), dim=1))
            real_sequence_reversed = self.embed(
                t.flip(t.cat((x, y), dim=1), dims=(1,)), future=False
            )
            fake_sequence_reversed = self.embed(
                t.cat((t.flip(y, dims=(1,)), fake_x), dim=1), future=False
            )
            self.real_sequence_detached = (
                real_sequence.detach()
            )  # used later by discriminators
            self.fake_sequence_detached = (
                fake_sequence.detach()
            )  # used later by discriminators
            self.real_sequence_reversed_detached = (
                real_sequence_reversed.detach()
            )
            self.fake_sequence_reversed_detached = (
                fake_sequence_reversed.detach()
            )
            future_generator_loss = self.one_way_generator_loss(
                fake_sequence, future=True
            )
            past_generator_loss = self.one_way_generator_loss(
                fake_sequence_reversed, future=False
            )
            cycle_loss = self.adversarial_loss(t.flip(fake_x, dims=(1,)), x)
            y_mse = F.mse_loss(fake_y, y)
            generator_loss = (
                future_generator_loss + past_generator_loss + cycle_loss
            ) / 3

            tqdm_dict = {"g_loss": generator_loss}
            # not used for backpropagation
            return {
                "loss": generator_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "train_mse": y_mse,
            }

        # train temporal discriminator
        if optimizer_idx == 2:
            future_temp_disc_loss = self.one_way_temporal_discriminator_loss(
                self.real_sequence_detached,
                self.fake_sequence_detached,
                future=True,
            )
            past_temp_disc_loss = self.one_way_temporal_discriminator_loss(
                self.real_sequence_reversed_detached,
                self.fake_sequence_reversed_detached,
                future=False,
            )
            temp_disc_loss = (future_temp_disc_loss + past_temp_disc_loss) / 2
            tqdm_dict = {"td_loss": temp_disc_loss}
            # self.log('td_loss', temp_disc_loss, prog_bar=True)
            return {
                "loss": temp_disc_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }

    def training_epoch_end(self, outputs):
        avg_loss = t.stack([x[0]["train_mse"] for x in outputs]).mean()
        td_loss = t.stack([x[2]["loss"] for x in outputs]).mean()
        self.log("train_mse", avg_loss, prog_bar=True)
        self.log("train_td", td_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.params.lr
        b1 = self.params.b1
        b2 = self.params.b2

        generator_optimizer = t.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        temporal_discriminator_optimizer = t.optim.Adam(
            self.temporal_discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        return (
            [generator_optimizer, temporal_discriminator_optimizer,],
            [],
        )
