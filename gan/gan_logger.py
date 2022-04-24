from pytorch_lightning.loggers import LightningLoggerBase
import torch as t
import json
import csv
import os
import ipdb
from argparse import Namespace


class GANLogger(LightningLoggerBase):
    def __init__(
        self,
        params: Namespace,
        train_header: tuple[str, ...] = ("epoch", "train_mse", "val_mse"),
    ):
        super().__init__()
        self.params = params
        self.csv_path = os.path.join(params.save_path, "training_log.csv")
        self.last_train_mse_loss = -1.0
        with open(self.csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(train_header)

    def log_metrics(self, metrics, step: int):
        if "train_mse" in metrics:
            self.last_train_mse_loss = metrics["train_mse"]
        elif "val_mse" in metrics:
            with open(self.csv_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    (metrics["epoch"], self.last_train_mse_loss, metrics["val_mse"])
                )
        elif "test_performance" in metrics:
            test_path = os.path.join(self.params.save_path, "test_performance.csv")
            metrics = (('model', self.params.model),) + tuple((key, val) for key, val in metrics['test_performance'].items())
            with open(test_path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(tuple(key for key, val in metrics))
                writer.writerow(tuple(val for key, val in metrics))

    @property
    def name(self):
        return "GANLogger"

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass
