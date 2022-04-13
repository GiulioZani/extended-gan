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


from .metrics import MetricsManager, IncrementalTuple


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

from .data_loader import get_loaders, DataLoader
from .utils import (
    get_number_of_params,
    visualize_predictions,
    IncrementalAccuracy,
    accuracy_criterion,
    TrainingHistory,
)


architecture_params = {
    "netG": ConvGenerator,
    "netFD": Conv3DFrameDiscriminator,
    "netTD": Conv3DTemporalDiscriminator,
}


Generator = architecture_params["netG"]
FrameDiscriminator = architecture_params["netFD"]
TemporalDiscriminator = architecture_params["netTD"]


# def gen_loss(fd_labels, td_labels):
#     loss =   (t.sum(t.log( t.ones_like(fd_labels) - fd_labels ) )) +  (t.sum(t.log(t.ones_like(td_labels) - td_labels)))

#     return loss #/ fd_labels.size(0)
def test(
    dataloader: DataLoader,
    netG: Generator,
    netFD: FrameDiscriminator,
    netTD: TemporalDiscriminator,
    device: t.device,
    epoch: int,
    params,
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
    denorm_denom = 0.0
    loss_model = 0.0
    denorm_loss_model = 0.0

    with t.no_grad():
        for i, (x, y) in enumerate(dataloader):
            # select the first n channels
            y = y[:, :, 0 : params["nc"], ...]
            x = x[:, :, 0 : params["nc"], ...]
            # all the past data, not all is passed to the generator (discriminator gets all)
            past_x = x
            x = x[
                :, params["in_seq_len"] - params["generator_in_seq_len"] :, ...
            ]

            y = t.sum(y.flatten() ** 2)
            b_size = x.size(0)
            real_label = t.zeros(b_size, 1, device=device) + 1
            fake_label = t.zeros(b_size, 1, device=device)

            if i == 0:
                pred_y = netG(x).cpu()
                visualize_predictions(x, y, pred_y, epoch, img_path)

            pred_real_frame_label = netFD(y)
            pred_real_temp_label = netTD(t.cat((x, y), dim=1))

            acc_FD_real = accuracy_criterion(pred_real_frame_label, real_label)
            acc_TD_real = accuracy_criterion(pred_real_temp_label, real_label)

            pred_y = netG(x).detach()
            pred_fake_frame_label = netFD(pred_y)
            pred_fake_temp_label = netTD(t.cat((past_x, pred_y), dim=1))

            # fd_metrics.update(pred_fake_frame_label, fake_label)
            # td_metrics.update(pred_fake_temp_label, fake_label)

            acc_FD_real = accuracy_criterion(pred_real_frame_label, real_label)
            acc_TD_real = accuracy_criterion(pred_real_temp_label, real_label)

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
                y_true_mask.flatten() * 2 + y_pred_mask.flatten(), minlength=4,
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
    nmse = loss_model / denorm_denom
    loss_model /= total_length

    netG.train()
    netTD.train()
    netFD.train()
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

curdir = os.path.dirname(__file__)


def train_single_epoch(
    *,
    dataloader: DataLoader,
    netG: Generator,
    netFD: FrameDiscriminator,
    netTD: TemporalDiscriminator,
    optimizerG,
    optimizerFD,
    optimizerTD,
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
        # ipdb.set_trace()
        # ipdb.set_trace()

        #       crop the channels that we dont use
        y = y[:, :, : params["nc"], ...]
        x = x[:, :, : params["nc"], ...]

        # y = y.squeeze(2)
        # x = x.squeeze(2)

        # Saving the original past data
        data_original = x

        #
        data = x[
            :, params["in_seq_len"] - params["generator_in_seq_len"] :, ...
        ]
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = data_original.size(0)
       
        # ipdb.set_trace()

        # Make accumalated gradients of the discriminator zero.
        netTD.zero_grad()
        netFD.zero_grad()
        # Create labels for the real data. (label=1)
        real_label = t.zeros(b_size,1, device=device) +1 
        fake_label = t.zeros(b_size,1, device=device)
        # ipdb.set_trace()
        pred_real_frame_label = netFD(y)
        # ipdb.set_trace()
        pred_real_temp_label = netTD(t.cat((data_original, y), dim=1))
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
        fake_data = netG(data)
        # print(fake_data.shape)
        # ipdb.set_trace()
        # As no gradients w.r.t. the generator parameters are to be
        # calculated, detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        fake_data_detached = fake_data.detach()
        pred_fake_frame_label = netFD(fake_data_detached)
        pred_fake_temp_label = netTD(
            t.cat((data_original, fake_data_detached), dim=1)
        )
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
        optimizerFD.step()
        optimizerTD.step()

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
        netG.zero_grad()
        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        pred_frame_label = netFD(fake_data)
        pred_temp_label = netTD(t.cat((data_original, fake_data), dim=1))

        # ipdb.set_trace()

        
        errG =  criterion(pred_temp_label, real_label) + criterion(pred_frame_label, real_label) #+  criterion(fake_data, y) 
   
        # if we have nash equilibrium, we don't want to add random guess to the loss
        # ipdb.set_trace()
        inc_acc_G += (
            inc_acc_FD.reciprocal() + inc_acc_TD.reciprocal()
        )  # + inc_acc_G_MSE

        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()
        errG.backward()

        # D_G_z2 = output.mean().item()
        # Update generator parameters.
        optimizerG.step()

        # inc_acc_G_MSE += accuracy_criterion(fake_data, y)

        # if real_loss_G.item() < 0.005:
        #     noise = noise + noise_step
        #     print('curriculum noise:', noise)

        # Check progress of training.
        if i % 50 == 0:

            t.save(netG.state_dict(), os.path.join(curdir, "models", "netG_epoch_" + str(epoch) + ".pth"))
            t.save(netFD.state_dict(), os.path.join(curdir, "models", "netFD_epoch_" + str(epoch) + ".pth"))
            t.save(netTD.state_dict(), os.path.join(curdir, "models", "netTD_epoch_" + str(epoch) + ".pth"))
            
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


def train():
    curdir = os.path.dirname(__file__)
    # Set random seed for reproducibility.
    seed = 369
    random.seed(seed)
    t.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    params = {
        "bsize": 8,  # Batch size during training.
        "imsize": 32,  # Spatial size of training images. All images will be resized to this size during preprocessing.
        "nc": 4,  # Number of channels
        "nz": 32,  # Size of the Z latent matrix (imsize * z)
        "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
        "ndf": 2,  # Size of features maps in the discriminator. The depth will be multiples of this.
        "nepochs": 10,  # Number of training epochs.
        "lr": 0.0002,  # Learning rate for optimizers
        "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
        "save_epoch": 2,
        "probalistic_gen": False,
        "in_seq_len": 8,
        "out_seq_len": 8,
        "generator_in_seq_len": 8,  # should be less than in_seq_len
        "add_gaussian_noise_to_gen": True,
        "gaussian_noise_std": 0.0005,  # std of gaussian noise added to the generator
        "debug": False,
    }

    # Use GPU is available else use CPU.
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    if params["debug"]:
        device = t.device("cpu")
    print(device, " will be used.\n")

    # Create the generator.
    netG = Generator(params).to(device)
    # netG.load_state_dict(t.load(os.path.join(curdir, 'models/netG.pth'), map_location=device))

    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2
    netG.apply(weights_init)
    # Print the model.
    print(netG)

    # Create the discriminator.
    netFD = FrameDiscriminator(params).to(device)
    netTD = TemporalDiscriminator(params).to(device)
    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2
    netFD.apply(weights_init)
    netTD.apply(weights_init)
    # Print the model.
    print(netTD)
    print(netFD)

    netTD.load_state_dict(t.load(os.path.join(curdir, 'models/netTD_epoch_1.pth'), map_location=device))
    netFD.load_state_dict(t.load(os.path.join(curdir, 'models/netFD_epoch_1.pth'), map_location=device))
    netG.load_state_dict(t.load(os.path.join(curdir, 'models/netG_epoch_1.pth'), map_location=device))

    params_dict = {
        "netG": get_number_of_params(netG),
        "netTD": get_number_of_params(netTD),
        "netFD": get_number_of_params(netFD),
    }
    print(params_dict)

    # netLTD = ConvTemporalDiscriminator(params).to(device)

    # netLTD.apply(weights_init)
    # print(netLTD)

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()

    # Optimizer for the discriminator.
    optimizerTD = optim.Adam(
        netTD.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )
    optimizerFD = optim.Adam(
        netFD.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )
    # Optimizer for the generator.
    optimizerG = optim.Adam(
        netG.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )
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
            netG=netG,
            netFD=netFD,
            netTD=netTD,
            optimizerG=optimizerG,
            optimizerFD=optimizerFD,
            optimizerTD=optimizerTD,
            criterion=criterion,
            device=device,
            epoch=epoch,
            params=params,
        )
        test_result = test(
            test_data_loader, netG, netFD, netTD, device, epoch, params
        )
        results = train_result | test_result
        print(json.dumps(results, indent=4))
        history.append(results)
        t.save(
            netG.state_dict(),
            os.path.join(
                curdir, "models", "netG_epoch_" + str(epoch) + ".pth"
            ),
        )
        t.save(
            netFD.state_dict(),
            os.path.join(
                curdir, "models", "netFD_epoch_" + str(epoch) + ".pth"
            ),
        )
        t.save(
            netTD.state_dict(),
            os.path.join(
                curdir, "models", "netTD_epoch_" + str(epoch) + ".pth"
            ),
        )

    history.plot()

    t.save(netG.state_dict(), os.path.join(curdir, "models", "netG.pth"))
    t.save(netFD.state_dict(), os.path.join(curdir, "models", "netFD.pth"))
    t.save(netTD.state_dict(), os.path.join(curdir, "models", "netTD.pth"))
