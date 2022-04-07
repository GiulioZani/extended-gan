import pytorch_lightning as pl
from .model import Generator, TemporalDiscriminator, FrameDiscriminator, weights_init
import torch.optim as optim
import torch as t


class ExtendendGAN(pl.LightningModule):
    def __init__(self, params: dict):
        self.params = params
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.generator = Generator(params).to(self.device)
        self.temp_disc = TemporalDiscriminator(params).to(self.device)
        self.frame_disc = FrameDiscriminator(params).to(self.device)

    def configure_optimizers(self):
        return [
            optim.Adam(
                self.generator.parameters(),
                lr=self.params["lr"],
                betas=(self.params["beta1"], 0.999),
            ),
            optim.Adam(
                self.temp_disc.parameters(),
                lr=self.params["lr"],
                betas=(self.params["beta1"], 0.999),
            ),
            optim.Adam(
                self.frame_disc.parameters(),
                lr=self.params["lr"],
                betas=(self.params["beta1"], 0.999),
            ),
        ]
