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
from .test import test
    


from .data_loader import get_loaders, DataLoader
from .utils import (
    visualize_predictions,
    IncrementalAccuracy,
    accuracy_criterion,
    TrainingHistory,
)


val_mse_validation_data = 10

noise = 0.001
noise_step = 0.005

def train_single_epoch(
    *,
    dataloader: DataLoader,
    gen,
    netFD,
    netTD,
    optimizerG,
    optimizerFD,
    optimizerTD,
    criterion,
    device: t.device,
    epoch: int,
):
    inc_acc_FD = IncrementalAccuracy()
    inc_acc_TD = IncrementalAccuracy()
    inc_acc_G = IncrementalAccuracy()
    inc_acc_G_MSE = IncrementalAccuracy()

    noise_epoch_added = 0

    for i, (x, y) in enumerate(dataloader):
        y = y.squeeze(2)
        data = x.squeeze(2)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = data.size(0)
        # ipdb.set_trace()

        # Make accumalated gradients of the discriminator zero.
        netTD.zero_grad()
        netFD.zero_grad()
        # Create labels for the real data. (label=1)
        real_label = t.zeros(b_size, device=device) + 1
        fake_label = t.zeros(b_size, device=device)

        pred_real_frame_label = netFD(y)
        pred_real_temp_label = netTD(t.cat((data, y), dim=1))
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
        pred_fake_frame_label = netFD(fake_data_detached)
        pred_fake_temp_label = netTD(t.cat((data, fake_data_detached), dim=1))
        errFD_fake = criterion(pred_fake_frame_label, fake_label)
        errTD_fake = criterion(pred_fake_temp_label, fake_label)
        inc_acc_FD += accuracy_criterion(pred_fake_frame_label, fake_label)
        inc_acc_TD += accuracy_criterion(pred_fake_temp_label, fake_label)

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

        # Make accumalted gradients of the generator zero.
        gen.zero_grad()
        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        pred_frame_label = netFD(fake_data).view(-1)
        pred_temp_label = netTD(t.cat((data, fake_data), dim=1)).view(-1)
        errG = criterion(pred_frame_label, real_label) + criterion(
            pred_temp_label, real_label
        ) 
        inc_acc_G += inc_acc_FD.reciprocal() + inc_acc_TD.reciprocal() #+ inc_acc_G_MSE
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()
        errG.backward()

        # D_G_z2 = output.mean().item()
        # Update generator parameters.
        optimizerG.step()

        real_loss_G = nn.MSELoss()(fake_data, y) 
        # inc_acc_G_MSE += accuracy_criterion(fake_data, y)

        # if real_loss_G.item() < 0.005:
        #     noise = noise + noise_step
        #     print('curriculum noise:', noise)

        # Check progress of training.
        if i % 50 == 0:
            print(
                # f"[{epoch}/{params['nepochs']}]\t"
                f"Loss_FD: {errFD.item():.4f}\t"
                + f"Loss_TD: {errTD.item():.4f}\t"
                + f"Loss_G: {errG.item():.4f}\t"
                + f"Loss_G_MSE: {real_loss_G.item():.4f}\t"

                # + f"Loss_G_ICN_MSE: {inc_acc_G_MSE.item():.4f}\t"
                # + f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.14f} / {D_G_z2:.4f}"
            )
    return {
        # "train_acc_temp_disc": inc_acc_TD.item(),
        # "train_acc_frame_disc": inc_acc_FD.item(),
        # "train_acc_gen": inc_acc_G.item(),
    }


def train(*, params, gen, netFD, netTD, optimizerG, optimizerFD, optimizerTD):
    curdir = os.path.dirname(__file__)
    # Set random seed for reproducibility.
    seed = 369
    random.seed(seed)
    t.manual_seed(seed)
    print("Random Seed: ", seed)

    # Parameters to define the model.
    

    # Use GPU is available else use CPU.
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(device, " will be used.\n")

    # Create the generator.
    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2


    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2
    netFD.apply(weights_init)
    netTD.apply(weights_init)
    # Print the model.
    print(netTD)
    print(netFD)

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()

    # Optimizer for the discriminator.
    
    history = TrainingHistory()

    for epoch in range(1, params["nepochs"] + 1):

        train_data_loader, test_data_loader = get_loaders(
            "./datasets/data", 32, 64, device, seq_len=params["nc"]
        )
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
        )
        test_result = test(test_data_loader, netG, netFD, netTD, device, epoch)
        results = train_result | test_result
        print(json.dumps(results, indent=4))
        history.append(results)
    history.plot()
