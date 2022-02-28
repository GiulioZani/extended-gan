import numpy as np
import torch as t
import ipdb
import enum
import matplotlib.pyplot as plt
import os


class TrainingHistory:
    def __init__(self, history=None):
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

    def plot(self, save=True, file_name="history"):
        plt.clf()
        for key, val in self.history.items():
            plt.plot(
                val,
                label=key.replace("_", " "),
            )
        plt.legend()
        plt.title("Training History")
        if save:
            plt.savefig(file_name)
        else:
            plt.show()
        plt.close()


def get_number_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def round_tensor(a: t.Tensor):
    return t.round(a).int() if a.dtype not in (t.long, t.int) else a


def accuracy_criterion(a: t.Tensor, b: t.Tensor):
    return IncrementalAccuracy(
        t.tensor([t.sum(round_tensor(a) == round_tensor(b)), a.shape[0]])
    )


class IncrementalAccuracy:
    def __init__(self, val=None):
        if val == None:
            self.val = t.tensor([0, 0])
        else:
            self.val = val

    def reciprocal(self):
        return IncrementalAccuracy(t.tensor([self.val[1] - self.val[0], self.val[1]]))

    def __add__(self, x):
        return IncrementalAccuracy(x.val + self.val)

    def __iadd__(self, x):
        self.val += x.val
        return self

    def item(self):
        return (self.val[0] / self.val[1]).item()

    def __str__(self):
        return f"{self.item()}"

    def __format__(self, x):
        return self.item().__format__(x)


def visualize_predictions(x, y, preds, epoch=1, path="", show_plot=False):
    if path != '' and not os.path.exists(path):
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


def get_metrics(y, y_hat, mean):
    y = t.clone(y.cpu())
    y_hat = t.clone(y_hat.cpu())
    y[y < mean] = 0
    y[y >= mean] = 1
    y_hat[y_hat < mean] = 0
    y_hat[y_hat >= mean] = 1
    acc = accuracy(y, y_hat)
    prec = precision(y, y_hat)
    # if prec == t.nan:
    #    ipdb.set_trace()
    rec = recall(y, y_hat)
    return acc, prec, rec


def accuracy(y, y_hat):
    return (y == y_hat).sum() / y[0].numel()


def precision(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    return (TP / (TP + FP)) * len(y_true)


def recall(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    # FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    result = (TP / (TP + FN)) * len(y_true)
    # jif result.isnan():
    #    ipdb.set_trace()
    return result


def denormalize(x, mean, var):
    mean = t.mean(mean)
    var = t.var(var)
    return x * var + mean
