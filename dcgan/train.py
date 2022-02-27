import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
import random
import os
import ipdb

from .model import (
    weights_init,
    Generator,
    FrameDiscriminator,
    TemporalDiscriminator,
)
from .data_loader import get_loaders, DataLoader
from .utils import visualize_predictions, IncrementalAccuracy, accuracy_criterion

def test(
    dataloader: DataLoader,
    netG: Generator,
    netFD: FrameDiscriminator,
    netTD: TemporalDiscriminator,
    device: t.device,
    epoch: int,
):
    netG.eval()
    netFD.eval()
    netTD.eval()
    
    inc_acc_FD = IncrementalAccuracy()
    inc_acc_TD = IncrementalAccuracy()
    inc_acc_G = IncrementalAccuracy()
    with t.no_grad():
        for i, (data, y) in enumerate(dataloader):
            y = y.squeeze(2)
            data = data.squeeze(2)
            real_data = data
            b_size = real_data.size(0)
            real_label = (t.zeros(b_size, device=device) + 1)
            fake_label = t.zeros(b_size, device=device)

            if i == 0:
                fake_data = netG(data).cpu()
                visualize_predictions(data, y, fake_data, epoch)

            pred_real_frame_label = netFD(y)
            pred_real_temp_label = netTD(t.cat((data, y), dim=1))
            errFD_real = accuracy_criterion(pred_real_frame_label, real_label)
            errTD_real = accuracy_criterion(pred_real_temp_label, real_label)

            fake_data = netG(data)
            fake_data_detached = fake_data.detach()
            pred_fake_frame_label = netFD(fake_data_detached)
            pred_fake_temp_label = netTD(
                t.cat((data, fake_data_detached), dim=1)
            )
            errFD_fake = accuracy_criterion(pred_fake_frame_label, fake_label)
            errTD_fake = accuracy_criterion(pred_fake_temp_label, fake_label)

            inc_acc_FD += errFD_real + errFD_fake
            inc_acc_TD += errTD_real + errTD_fake

            inc_acc_G += inc_acc_FD.reciprocal() + inc_acc_TD.reciprocal()
    print(f"{inc_acc_FD=:.4f} {inc_acc_TD=:.4f} {inc_acc_G=:.4f}")
    netG.train()
    netTD.train()
    netFD.train()
    return {
        'acc_temp_disc':inc_acc_TD.item(),
        'acc_frame_disc':inc_acc_TD.item(),
        'acc_gen':inc_acc_TD.item()
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
        "bsize": 128,  # Batch size during training.
        "imsize": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
        "nc": 6,  # Number of channles in the training images. For coloured images this is 3.
        "nz": 100,  # Size of the Z latent vector (the input to the generator).
        "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
        "ndf": 64,  # Size of features maps in the discriminator. The depth will be multiples of this.
        "nepochs": 10,  # Number of training epochs.
        "lr": 0.0002,  # Learning rate for optimizers
        "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
        "save_epoch": 2,
    }  # Save step.

    # Use GPU is available else use CPU.
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(device, " will be used.\n")

    # Get the data.
    dataloader, test_data_loader = get_loaders(
        "./datasets/data", 64, 64, device, seq_len=params["nc"]
    )

    # Create the generator.
    netG = Generator(params).to(device)
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

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()

    fixed_noise = t.randn(64, params["nz"], 1, 1, device=device)

    real_label = 1
    fake_label = 0

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

    iters = 0
    print("Starting Training Loop...")
    print("-" * 25)

    for epoch in range(1, params["nepochs"] + 1):
        dataloader, test_data_loader = get_loaders(
            "./datasets/data", 32, 64, device, seq_len=params["nc"]
        )
        for i, (data, y) in enumerate(dataloader):

            y = y.squeeze(2)
            data = data.squeeze(2)
            # Transfer data tensor to GPU/CPU (device)
            real_data = data
            # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
            b_size = real_data.size(0)

            # Make accumalated gradients of the discriminator zero.
            netTD.zero_grad()
            netFD.zero_grad()
            # Create labels for the real data. (label=1)
            label = t.full((b_size,), real_label, device=device).float()
            pred_frame_label = netFD(y)
            pred_temp_label = netTD(t.cat((data, y), dim=1))
            errFD_real = criterion(pred_frame_label, label)
            errTD_real = criterion(pred_temp_label, label)
            # Calculate gradients for backpropagation.
            errFD_real.backward()
            errTD_real.backward()
            # D_x = output.mean().item()

            # Sample random data from a unit normal distribution.
            # noise = torch.randn(b_size, params["nz"], 1, 1, device=device)
            # Generate fake data (images).
            fake_data = netG(data)
            # Create labels for fake data. (label=0)
            label[:] = fake_label
            # Calculate the output of the discriminator of the fake data.
            # As no gradients w.r.t. the generator parameters are to be
            # calculated, detach() is used. Hence, only gradients w.r.t. the
            # discriminator parameters will be calculated.
            # This is done because the loss functions for the discriminator
            # and the generator are slightly different.
            fake_data_detached = fake_data.detach()
            pred_frame_label = netFD(fake_data_detached)
            pred_temp_label = netTD(
                t.cat((data, fake_data_detached), dim=1)
            )
            errFD_fake = criterion(pred_frame_label, label)
            errTD_fake = criterion(pred_temp_label, label)
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
            netG.zero_grad()
            # We want the fake data to be classified as real. Hence
            # real_label are used. (label=1)
            label[:] = real_label
            # No detach() is used here as we want to calculate the gradients w.r.t.
            # the generator this time.
            pred_frame_label = netFD(fake_data).view(-1)
            pred_temp_label = netTD(t.cat((data, fake_data), dim=1)).view(-1)
            errG = criterion(pred_frame_label, label) + criterion(
                pred_temp_label, label
            )
            # Gradients for backpropagation are calculated.
            # Gradients w.r.t. both the generator and the discriminator
            # parameters are calculated, however, the generator's optimizer
            # will only update the parameters of the generator. The discriminator
            # gradients will be set to zero in the next iteration by netD.zero_grad()
            errG.backward()

            # D_G_z2 = output.mean().item()
            # Update generator parameters.
            optimizerG.step()

            # Check progress of training.
            if i % 50 == 0:
                print(
                    f"[{epoch}/{params['nepochs']}]\t"
                    + f"Loss_FD: {errFD.item():.4f}\t"
                    + f"Loss_TD: {errTD.item():.4f}\t"
                    + f"Loss_G: {errG.item():.4f}\t"
                    # + f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.14f} / {D_G_z2:.4f}"
                )

            # Save the losses for plotting.
            G_losses.append(errG.item())
            D_losses.append((errFD.item(), errTD.item()))

            # Check how the generator is doing by saving G's output on a fixed noise.
            if iters % 50 == 0:
                with t.no_grad():
                    fake_data = netG(data).detach().cpu()
                    visualize_predictions(data, y, fake_data, epoch)
            iters += 1

        test(test_data_loader, netG, netFD, netTD, device, epoch)
        """
        # Save the model.
        if epoch % params["save_epoch"] == 0:
            t.save(
                {
                    "generator": netG.state_dict(),
                    "discriminator": netD.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "params": params,
                },
                os.path.join(curdir, "model/model_epoch_{}.pth".format(epoch)),
            )
        """

    """
    # Save the final trained model.
    torch.save(
        {
            "generator": netG.state_dict(),
            "discriminator": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
            "params": params,
        },
        os.path.join(curdir, "model/model_final.pth"),
    )
    """
