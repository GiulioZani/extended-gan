import numpy as np
import torch as t
import ipdb
import enum
import matplotlib.pyplot as plt
import os


def get_number_parameters(model: t.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def visualize_predictions(
    x: t.Tensor,
    y: t.Tensor,
    preds: t.Tensor,
    epoch=1,
    path="imgs/",
):
    if not os.path.exists(path):
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

    save_path = os.path.join(path, f"pred_{epoch}.png")
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_history(
    history: dict[str, list[float]],
    title: str = "Training History",
    save=False,
    filename="history",
):
    plt.clf()
    plt.plot(
        history["train_loss"],
        label="Train loss",
    )
    plt.plot(
        history["val_loss"],
        label="Val loss",
    )
    plt.legend()
    plt.title(title)
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


def update_history(history: dict[str, list[float]], data: dict[str, float]):
    for key, val in data.items():
        if key not in history:
            history[key] = []
        history[key].append(val)


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

