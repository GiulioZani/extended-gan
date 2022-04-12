import imp
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
import random
import os
import ipdb
import json
from .data_loader import get_loaders, DataLoader
from .utils import (
    visualize_predictions,
    IncrementalAccuracy,
    accuracy_criterion,
    TrainingHistory,
)


def test(
    dataloader,
    netG,
    netFD,
    netTD,
    device: t.device,
    epoch: int,
):
    img_path = os.path.join(os.path.dirname(__file__), "imgs")

    netG.eval()
    netFD.eval()
    netTD.eval()

    inc_acc_FD = IncrementalAccuracy()
    inc_acc_TD = IncrementalAccuracy()
    inc_acc_G = IncrementalAccuracy()
    total_length = 0
    threshold = 0.5
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    loss_model = 0.0
    denorm_loss_model = 0.0

    with t.no_grad():
        for i, (x, y) in enumerate(dataloader):
            y = y.squeeze(2)
            x = x.squeeze(2)
            real_data = x
            b_size = real_data.size(0)
            real_label = t.zeros(b_size, device=device) + 1
            fake_label = t.zeros(b_size, device=device)

            if i == 0:
                pred_y = netG(x).cpu()
                visualize_predictions(x, y, pred_y, epoch, img_path)

            pred_real_frame_label = netFD(y)
            pred_real_temp_label = netTD(t.cat((x, y), dim=1))
            acc_FD_real = accuracy_criterion(pred_real_frame_label, real_label)
            acc_TD_real = accuracy_criterion(pred_real_temp_label, real_label)

            pred_y = netG(x).detach()
            pred_fake_frame_label = netFD(pred_y)
            pred_fake_temp_label = netTD(
                t.cat((x, pred_y), dim=1)
            )
            acc_FD_fake = accuracy_criterion(pred_fake_frame_label, fake_label)
            acc_TD_fake = accuracy_criterion(pred_fake_temp_label, fake_label)

            inc_acc_FD += acc_FD_real + acc_FD_fake
            inc_acc_TD += acc_TD_real + acc_TD_fake
            inc_acc_G += inc_acc_FD.reciprocal() + inc_acc_TD.reciprocal()
            loss_model += t.nn.functional.mse_loss(
                pred_y.squeeze(), y.squeeze(), reduction="sum"
            )
            y_true_mask = y > threshold
            y_pred_mask = pred_y > threshold
            tn, fp, fn, tp = t.bincount(
                y_true_mask.flatten() * 2 + y_pred_mask.flatten(),
                minlength=4,
            )
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            total_length += y.numel()
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    accuracy = (total_tp + total_tn) / (
        total_tp + total_tn + total_fp + total_fn
    )
    f1 = 2 * precision * recall / (precision + recall)
    loss_model /= total_length

    netG.train()
    netTD.train()
    netFD.train()
    return {
        "Temp. Disc. Accuracy": inc_acc_TD.item(),
        "Frame Dis. Accuracy": inc_acc_FD.item(),
        "MSE": loss_model,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "f1": f1,
    }

