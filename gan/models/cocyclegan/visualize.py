import os
import threading
import torch as t
import matplotlib.pyplot as plt


def denorm(x):
    return (x + 1) / 2


def run_on_thread(func, *args, **kwargs):
    t = threading.Thread(target=func, args=args, kwargs=kwargs)
    t.start()
    return t


def visualize_predictions(*args, **kwargs):
    run_on_thread(_visualize_predictions, *args, **kwargs)


def _visualize_predictions(
    x: t.Tensor,
    y: t.Tensor,
    preds: t.Tensor,
    gt_reverse: t.Tensor,
    pred_reverse: t.Tensor,
    epoch=1,
    path="imgs/",
    flag="train",
):
    if not os.path.exists(path):
        os.mkdir(path)
    to_plot = [
        denorm(x[0].squeeze(1)),
        denorm(y[0].squeeze(1)),
        denorm(preds[0].squeeze(1)),
        denorm(gt_reverse[0].squeeze(1)),
        denorm(pred_reverse[0].squeeze(1)),
    ]

    _, ax = plt.subplots(nrows=len(to_plot), ncols=to_plot[0].shape[0])
    plt.suptitle(f"Epoch {epoch}")
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            uba = to_plot[i].cpu().detach().numpy()[j]
            col.imshow(uba)

    row_labels = ["input", "GT", "pred", "GT R", "Pred R"]
    for ax_, row in zip(ax[:, 0], row_labels):
        ax_.set_ylabel(row)

    col_labels = [f"F{i}" for i in range(to_plot[0].shape[0])]
    for ax_, col in zip(ax[0, :], col_labels):
        ax_.set_title(col)

    # remove ticks from all but the bottom row
    for ax_ in ax.flat:
        ax_.xaxis.set_ticklabels([])
        ax_.yaxis.set_ticklabels([])

    save_path = os.path.join(path, f"{flag}_{epoch}.png")
    plt.savefig(save_path)
    # plt.show()
    plt.close()
