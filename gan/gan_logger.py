from pytorch_lightning.loggers import LightningLoggerBase
import torch as t
import json
import csv
import os
import ipdb

class GANLogger(LightningLoggerBase):
    def __init__(
        self,
        loggin_path: str,
        train_header: tuple[str, ...] = ("epoch", "train_mse", "val_mse"),
    ):
        super().__init__()
        self.csv_path = os.path.join(loggin_path, "training_log.csv")
        self.last_train_mse_loss = -1.0
        self.test_metrics = {}
        with open(self.csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(train_header)

    def log_metrics(self, metrics, step):
        if "train_mse" in metrics:
            self.last_train_mse_loss = metrics["train_mse"]
        elif "val_mse" in metrics:
            with open(self.csv_path, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    (metrics["epoch"], self.last_train_mse_loss, metrics["val_mse"])
                )
        else:
            self.test_metrics.update(metrics)
            ipdb.set_trace()

    @property
    def name(self):
        return "GANLogger"

    def log_hyperparams(self, hparams):
        pass

    @property
    def version(self):
        pass
