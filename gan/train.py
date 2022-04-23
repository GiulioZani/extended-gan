import imp
import math
from platform import architecture
from unittest import result
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

from .test import test
from .base_model import BaseModel
from .metrics import MetricsManager, IncrementalTuple
from .data_loader import get_loaders, DataLoader
from .utils import (
    get_number_of_params,
    visualize_predictions,
    IncrementalAccuracy,
    accuracy_criterion,
    TrainingHistory,
)

"""
from .conv2dmodel import (
    FrameDiscriminator,
    TemporalDiscriminator,
    weights_init,
    # ProbablisticConvGenerator as Generator,
    # FrameDiscriminator,
    # TemporalDiscriminator
)
from .resnetmodel import (
    ResNetAutoEncoder as Generator,
    VAE as AAGenerator,
    ResNetTemproalDiscriminator,
    ResNetFrameDiscriminator,
)

from .lstmconvmodel import (
    EncoderDecoderConvLSTM as LSTMGenerator,
    ConvLSTMTemporalDiscriminator as LSTMTemporalDiscriminator,
)

from .conv3dmodel import (
    Conv3DFrameDiscriminator,
    Conv3DTemporalDiscriminator,
    ConvGenerator,
)
from .resnet3d import (
    ResNet3DClassifier as ResNet3DFrameDiscriminator,
    ResNet3DAutoEncoder,
)
architecture_params = {
    "netG": ConvGenerator,
    "netFD": Conv3DFrameDiscriminator,
    "netTD": Conv3DTemporalDiscriminator,
}
Generator = architecture_params["netG"]
FrameDiscriminator = architecture_params["netFD"]
TemporalDiscriminator = architecture_params["netTD"]
"""

# def gen_loss(fd_labels, td_labels):
#     loss =   (t.sum(t.log( t.ones_like(fd_labels) - fd_labels ) )) +  (t.sum(t.log(t.ones_like(td_labels) - td_labels)))

#     return loss #/ fd_labels.size(0)

curdir = os.path.dirname(__file__)


def gan_predict_and_backward(
    *,
    x: t.Tensor,
    y: t.Tensor,
    gen: nn.Module,
    frame_disc: nn.Module,
    temp_disc: nn.Module,
    params,
    device: t.device,
    frame_disc_optim: optim.Optimizer,
    temp_disc_optim: optim.Optimizer,
    gen_optim: optim.Optimizer,
):
    # Saving the original past data
    data_original = x

    #
    data = x[:, params.in_seq_len - params.generator_in_seq_len :, ...]
    # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
    b_size = data_original.size(0)

    # ipdb.set_trace()

    # Make accumalated gradients of the discriminator zero.
    temp_disc.zero_grad()
    frame_disc.zero_grad()
    # Create labels for the real data. (label=1)
    real_label = t.zeros(b_size, 1, device=device) + 1
    fake_label = t.zeros(b_size, 1, device=device)
    # ipdb.set_trace()
    pred_real_frame_label = frame_disc(y)
    # ipdb.set_trace()
    pred_real_temp_label = temp_disc(t.cat((data_original, y), dim=1))
    # ipdb.set_trace()

    errFD_real = criterion(pred_real_frame_label, real_label)
    errTD_real = criterion(pred_real_temp_label, real_label)
    inc_acc_FD += accuracy_criterion(pred_real_frame_label, real_label)
    inc_acc_TD += accuracy_criterion(pred_real_temp_label, real_label)
    # inc_acc_G_MSE += accuracy_criterion(fake_data, y)
    # Calculate gradients for backpropagation.
    errFD_real.backward()
    errTD_real.backward()

    # Sample random data from a unit normal distribution.
    # noise = torch.randn(b_size, params["nz"], 1, 1, device=device)
    # Generate fake data (images).
    # global noise, val_mse_validation_data
    # if val_mse_validation_data < 0.008 and noise_epoch_added != epoch:
    #     noise = noise + noise_step
    #     print('curriculum noise:', noise)
    #     noise_epoch_added = epoch
    fake_data = gen(data)
    # print(fake_data.shape)
    # ipdb.set_trace()
    # As no gradients w.r.t. the generator parameters are to be
    # calculated, detach() is used. Hence, only gradients w.r.t. the
    # discriminator parameters will be calculated.
    # This is done because the loss functions for the discriminator
    # and the generator are slightly different.
    fake_data_detached = fake_data.detach()
    pred_fake_frame_label = frame_disc(fake_data_detached)
    pred_fake_temp_label = temp_disc(t.cat((data_original, fake_data_detached), dim=1))
    errFD_fake = criterion(pred_fake_frame_label, fake_label)
    errTD_fake = criterion(pred_fake_temp_label, fake_label)
    inc_acc_FD += accuracy_criterion(pred_fake_frame_label, fake_label)
    inc_acc_TD += accuracy_criterion(pred_fake_temp_label, fake_label)

    pred_metrics.update(y, fake_data)

    # Calculate gradients for backpropagation.
    errFD_fake.backward()
    errTD_fake.backward()
    # D_G_z1 = output.mean().item()

    # Net discriminator loss.
    errFD = errFD_real + errFD_fake
    errTD = errTD_real + errTD_fake
    # Update discriminator parameters.
    frame_disc_optim.step()
    temp_disc_optim.step()

    diff_square = (fake_data.flatten() - y.flatten()) ** 2
    running_mse += IncrementalAccuracy(
        t.tensor([diff_square.sum(), diff_square.numel()], dtype=t.float)
    )
    real_loss_G = nn.MSELoss()(fake_data, y)

    fd_metrics.update(pred_real_frame_label, real_label)
    fd_metrics.update(pred_fake_frame_label, fake_label)
    td_metrics.update(pred_real_temp_label, real_label)
    td_metrics.update(pred_fake_temp_label, fake_label)

    # Make accumalted gradients of the generator zero.
    gen.zero_grad()
    # We want the fake data to be classified as real. Hence
    # real_label are used. (label=1)
    # No detach() is used here as we want to calculate the gradients w.r.t.
    # the generator this time.
    pred_frame_label = frame_disc(fake_data)
    pred_temp_label = temp_disc(t.cat((data_original, fake_data), dim=1))

    # ipdb.set_trace()

    errG = criterion(pred_temp_label, real_label) + criterion(
        pred_frame_label, real_label
    )  # +  criterion(fake_data, y)

    # if we have nash equilibrium, we don't want to add random guess to the loss
    # ipdb.set_trace()
    inc_acc_G += inc_acc_FD.reciprocal() + inc_acc_TD.reciprocal()  # + inc_acc_G_MSE

    # Gradients for backpropagation are calculated.
    # Gradients w.r.t. both the generator and the discriminator
    # parameters are calculated, however, the generator's optimizer
    # will only update the parameters of the generator. The discriminator
    # gradients will be set to zero in the next iteration by netD.zero_grad()
    errG.backward()

    # D_G_z2 = output.mean().item()
    # Update generator parameters.
    gen_optim.step()


def train_single_epoch(
    *,
    dataloader: DataLoader,
    model: BaseModel,
    criterion,
    device: t.device,
    epoch: int,
    params: dict,
):
    td_metrics = MetricsManager(("precision",), prefix="temp_disc")
    inc_acc_FD = IncrementalAccuracy()
    inc_acc_TD = IncrementalAccuracy()
    inc_acc_G = IncrementalAccuracy()
    running_mse = IncrementalAccuracy()
    pred_metrics = MetricsManager(("mse",), prefix="train")
    fd_metrics = MetricsManager(("precision",), prefix="frame_disc")

    img_path = os.path.join(os.path.dirname(__file__), "validation")
    noise_epoch_added = 0

    for i, (x, y) in enumerate(dataloader):
        #       crop the channels that we dont use
        y = y[:, :, : params["nc"], ...]
        x = x[:, :, : params["nc"], ...]
        model.training_step(x, y)
        # y = y.squeeze(2)
        # x = x.squeeze(2)

        # inc_acc_G_MSE += accuracy_criterion(fake_data, y)

        # if real_loss_G.item() < 0.005:
        #     noise = noise + noise_step
        # Check progress of training.
        if i % 50 == 0:

            fake_data = netG(data).cpu()
            visualize_predictions(data, y, fake_data, epoch, img_path)
            print(
                # f"[{epoch}/{params['nepochs']}]\t"
                f"Loss_FD: {errFD.item():.4f}\t"
                + f"Loss_TD: {errTD.item():.4f}\t"
                + f"Loss_G: {errG.item():.4f}\t"
                + f"Loss_G_MSE: {real_loss_G.item():.8f}\t"
                # + f"Loss_G_MSE: {real_loss_G.item():.4f}\t"
                # + f"Loss_G_MSE: {real_loss_G.item():.4f}\t"
                # + f"Loss_G_ICN_MSE: {inc_acc_G_MSE.item():.4f}\t"
                # + f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.14f} / {D_G_z2:.4f}"
            )
    return {
        # "train_acc_temp_disc": inc_acc_TD.item(),
        # "train_acc_frame_disc": inc_acc_FD.item(),
        # "train_acc_gen": inc_acc_G.item(),
    }


def train(*, params, model: BaseModel):
    curdir = os.path.dirname(__file__)
    # Set random seed for reproducibility.
    seed = 369
    random.seed(seed)
    t.manual_seed(seed)
    print("Random Seed: ", seed)

    # Use GPU is available else use CPU.
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    if params["debug"]:
        device = t.device("cpu")
    print(device, " will be used.\n")

    # Create the generator.
    # gen = Generator(params).to(device)
    # netG.load_state_dict(t.load(os.path.join(curdir, 'models/netG.pth'), map_location=device))

    # Print the model.

    # netLTD = ConvTemporalDiscriminator(params).to(device)

    # netLTD.apply(weights_init)
    # print(netLTD)

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()

    # Optimizer for the discriminator.
    history = TrainingHistory()

    for epoch in range(1, params["nepochs"] + 1):

        train_data_loader, test_data_loader = get_loaders(
            "./datasets/data",
            params["bsize"],
            params["bsize"],
            device,
            crop=params["imsize"],
            in_seq_len=params["in_seq_len"],
            out_seq_len=params["out_seq_len"],
        )
        # test_result = test(test_data_loader, netG, netFD, netTD, device, epoch, params)
        # results = test_result
        # print(json.dumps(results, indent=4))
        train_result = train_single_epoch(
            dataloader=train_data_loader,
            model=model,
            criterion=criterion,
            device=device,
            epoch=epoch,
            params=params,
        )
        test_result = test(
            test_data_loader, gen, frame_disc, temp_disc, device, epoch, params
        )
        results = train_result | test_result
        print(json.dumps(results, indent=4))
        history.append(results)
        history.plot()

    t.save(gen.state_dict(), os.path.join(curdir, "models", "netG.pth"))
    t.save(frame_disc.state_dict(), os.path.join(curdir, "models", "netFD.pth"))
    t.save(temp_disc.state_dict(), os.path.join(curdir, "models", "netTD.pth"))
