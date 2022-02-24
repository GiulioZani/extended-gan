import torch as t
from argparse import ArgumentParser
import json

from .data_loader import get_loaders, DataLoader
from .models import Generator, Discriminator


def compute_loss(
    y_hat: t.Tensor, gen: Generator, disc: Discriminator
) -> t.Tensor:
    pass


def test(
    generator: Generator, discriminator: Discriminator, dl: DataLoader
) -> t.Tensor:
    discriminator.eval()
    generator.eval()
    with t.no_grad():
        for (x, y) in dl:
            pass
    discriminator.train()
    generator.train()


def train_single_epoch(
    gen: Generator,
    gen_optim: t.optim.Adam,
    disc: Discriminator,
    disc_optim: t.optim.Adam,
    dl: DataLoader,
) -> t.Tensor:
    gen.train()
    disc.train()
    for (x, y) in dl:
        y_hat = gen(x)
        loss = compute_loss(y_hat, gen, disc)
        loss.backward()
        gen_optim.step()
        disc_optim.step()


def train(
    epochs: int, data_folder: str, train_batch_size: int, test_batch_size: int
):
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen_optim = t.optim.Adam(gen.parameters())
    disc_optim = t.optim.Adam(disc.parameters())
    history = []
    for epoch in range(epochs):
        train_dl, test_dl = get_loaders(
            data_folder, train_batch_size, test_batch_size, device
        )
        train_loss = (
            train_single_epoch(gen, gen_optim, disc, disc_optim, train_dl)
            .cpu()
            .item()
        )
        test_loss = test(gen, disc, test_dl).cpu().item()
        print(
            f"Epoch: {epoch} Train loss: {train_loss:.6f} \n Val loss: {test_loss:.6f}"
        )
        history.append((train_loss, test_loss))


def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--data-folder", type=str, default="datasets/data",
    )
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))
    if args.action == "train":
        train(args.epochs, args.data_folder, 32, 64)
