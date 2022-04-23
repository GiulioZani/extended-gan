import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import ipdb


class GANLightning(LightningModule):
    def __init__(
        self, lr: float = 0.0002, b1: float = 0.5, b2: float = 0.999,
    ):
        super().__init__()

        # networks
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.generator = nn.Sequential()
        self.frame_discriminator = nn.Sequential()
        self.temporal_discriminator = nn.Sequential()

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
            ipdb.set_trace()
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
            return {"loss": generator_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}

        # train frame discriminator
        if optimizer_idx == 1:
            # Create labels for the real data. (label=1)
            # x_frames = x.view(batch_size * x_seq_len, channels, height, width)
            y_frames = y.view(batch_size * y_seq_len, channels, height, width)
            fake_frames = (
                self(x).detach().view(batch_size * y_seq_len, channels, height, width)
            )
            y_batch_size = batch_size * y_seq_len
            real_label = t.ones(y_batch_size, 1, device=self.device)
            fake_label = t.zeros(y_batch_size, 1, device=self.device)
            labels = t.cat((real_label, fake_label))
            frames = t.cat((y_frames, fake_frames))
            frame_disc_loss = self.adversarial_loss(
                self.frame_discriminator(frames), labels
            )
            tqdm_dict = {"frame_disc_loss": frame_disc_loss}
            return {
                "loss": frame_disc_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }

        # train temporal discriminator
        if optimizer_idx == 2:
            real_label = t.ones(batch_size, 1, device=self.device)
            fake_label = t.zeros(batch_size, 1, device=self.device)
            labels = t.cat((real_label, fake_label))
            fake_sequence = t.cat((x, self(x).detach()), dim=1)
            real_sequence = t.cat((x, y), dim=1)
            sequences = t.cat((real_sequence, fake_sequence))
            pred_labels = self.temporal_discriminator(sequences)
            temp_disc_loss = self.adversarial_loss(pred_labels, labels)
            tqdm_dict = {"temp_disc_loss": temp_disc_loss}
            return {
                "loss": temp_disc_loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            }

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

    def on_epoch_end(self):
        pass
