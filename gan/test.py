import imp
import torch as t
import torch.nn as nn
import os

# from .data_loader import get_loaders, DataLoader
from .utils import (
    visualize_predictions,
    IncrementalAccuracy,
    accuracy_criterion,
    TrainingHistory,
)


def gan_val_step(
    *,
    gen: nn.Module,
    fram_disc: nn.Module,
    temp_disc: nn.Module,
    x: t.Tensor,
    y: t.Tensor,
    device: t.device,
):
    b_size = x.size(0)
    real_label = t.zeros(b_size, 1, device=device) + 1
    fake_label = t.zeros(b_size, 1, device=device)

    if i == 0:
        pred_y = gen(x).cpu()
        visualize_predictions(x, y, pred_y, epoch, img_path)

    pred_real_frame_label = frame_disc(y)
    pred_real_temp_label = temp_disc(t.cat((x, y), dim=1))

    acc_FD_real = accuracy_criterion(pred_real_frame_label, real_label)
    acc_TD_real = accuracy_criterion(pred_real_temp_label, real_label)

    pred_y = gen(x).detach()
    pred_fake_frame_label = frame_disc(pred_y)
    pred_fake_temp_label = temp_disc(t.cat((past_x, pred_y), dim=1))

    # fd_metrics.update(pred_fake_frame_label, fake_label)
    # td_metrics.update(pred_fake_temp_label, fake_label)

    acc_FD_real = accuracy_criterion(pred_real_frame_label, real_label)
    acc_TD_real = accuracy_criterion(pred_real_temp_label, real_label)

    acc_FD_fake = accuracy_criterion(pred_fake_frame_label, fake_label)
    acc_TD_fake = accuracy_criterion(pred_fake_temp_label, fake_label)

    inc_acc_FD += acc_FD_real + acc_FD_fake
    inc_acc_TD += acc_TD_real + acc_TD_fake
    inc_acc_G += inc_acc_FD.reciprocal() + inc_acc_TD.reciprocal()
    return pred_y, inc_acc_FD, inc_acc_TD


def test(
    *, dataloader, model, params, epoch: int = -1, device: t.device,
):
    img_path = os.path.join(os.path.dirname(__file__), "imgs")

    gen.eval()
    frame_disc.eval()
    temp_disc.eval()

    inc_acc_FD = IncrementalAccuracy()
    inc_acc_TD = IncrementalAccuracy()
    inc_acc_G = IncrementalAccuracy()
    total_length = 0
    threshold = 0.5
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    denorm_denom = 0.0
    loss_model = 0.0

    with t.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # select the first n channels
            y = y[:, :, 0 : params["nc"], ...]
            x = x[:, :, 0 : params["nc"], ...]
            # all the past data, not all is passed to the generator (discriminator gets all)
            past_x = x
            x = x[:, params["in_seq_len"] - params["generator_in_seq_len"] :, ...]

            denorm_denom += t.sum(y.flatten() ** 2)

            loss_model += t.nn.functional.mse_loss(
                pred_y.squeeze(), y.squeeze(), reduction="sum"
            )
            y_true_mask = y > threshold
            y_pred_mask = pred_y > threshold
            tn, fp, fn, tp = t.bincount(
                y_true_mask.flatten() * 2 + y_pred_mask.flatten(), minlength=4,
            )
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            total_length += y.numel()
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)
    nmse = loss_model / denorm_denom
    loss_model /= total_length

    gen.train()
    temp_disc.train()
    frame_disc.train()
    # val_mse_validation_data = running_mse.item()
    return {
        "Temp. Disc. Accuracy": inc_acc_TD.item(),
        "Frame Dis. Accuracy": inc_acc_FD.item(),
        "MSE": float(loss_model),
        "Precision": float(precision),
        "Recall": float(recall),
        "Accuracy": float(accuracy),
        "NMSE": nmse.item(),
        "f1": float(f1),
    }
