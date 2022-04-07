import numpy as np
import torch as t
import ipdb
import enum
import matplotlib.pyplot as plt
import os
from .metrics import IncrementalTuple


class TrainingHistory:
    def __init__(
        self,
        history=None,
        groups: tuple[str, ...] = ("mse", "disc", ""),
        save_path: str = "training_plots",
    ):
        self.save_path = save_path
        self.groups = groups
        if history != None:
            self.history = history
        else:
            self.history = {}

    def append(self, new_data: dict):
        for key, val in new_data.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(val)
        return self

    def plot(self, save=True):
        keys = list(self.history.keys())
        used_keys = []
        key_groups = []
        for group_key in self.groups:
            group = []
            key_groups.append(group)
            for key in keys:
                if key not in used_keys:
                    if group_key in key:
                        group.append(key)
                        used_keys.append(key)
        for i, group_key in enumerate(key_groups):
            plt.clf()
            for key in group_key:
                plt.plot(
                    self.history[key], label=key.replace("_", " "),
                )
            plt.legend()
            plt.title("Training History")
            if save:
                plt.savefig(
                    os.path.join(
                        self.save_path,
                        f"{self.groups[i] if self.groups[i] != '' else 'metrics'}",
                    )
                )
            else:
                plt.show()
            plt.close()


def get_number_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def round_tensor(a: t.Tensor):
    return t.round(a).int() if a.dtype not in (t.long, t.int) else a


def accuracy_criterion(a: t.Tensor, b: t.Tensor):
    return IncrementalTuple(
        t.tensor([t.sum(round_tensor(a) == round_tensor(b)), a.shape[0]])
    )


def visualize_predictions(x, y, preds, epoch=1, path="", show_plot=False):
    if path != "" and not os.path.exists(path):
        os.mkdir(path)
    to_plot = [x[0], y[0].squeeze(1), preds[0]]
    _, ax = plt.subplots(nrows=len(to_plot), ncols=to_plot[0].shape[0])
    plt.suptitle(f"Epoch {epoch}")
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            uba = to_plot[i].cpu().detach().numpy()[j]
            col.imshow(uba)

    row_labels = ["input", "GT", "pred"]
    for ax_, row in zip(ax[:, 0], row_labels):
        ax_.set_ylabel(row)

    col_labels = [f"F{i}" for i in range(to_plot[0].shape[0])]
    for ax_, col in zip(ax[0, :], col_labels):
        ax_.set_title(col)

    save_path = os.path.join(path, f"pred.png")
    if not show_plot:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
