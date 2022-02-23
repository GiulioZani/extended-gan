import torch as t
from argparse import ArgumentParser
import json


def compute_loss():
    pass


def test():
    pass


def train_single_epoch():
    pass


def train(epochs: int, data_file_name: str):
    for epoch in range(epochs):
        train_single_epoch()
        test()


def main():
    parser = ArgumentParser()
    parser.add_argument("action", choices=("train", "test"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--data-file-name",
        type=str,
        default="datasets/coastal_preprocessed.pt",
    )
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))
    if args.action == "train":
        train(args.epochs, args.data_file_name)
