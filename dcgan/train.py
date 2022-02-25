import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
import random
import os
import ipdb

from .model import weights_init, Generator, Discriminator
from .data_loader import get_loaders
from .utils import visualize_predictions


def train():
    curdir = os.path.dirname(__file__)
    # Set random seed for reproducibility.
    seed = 369
    random.seed(seed)
    torch.manual_seed(seed)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    netD = Discriminator(params).to(device)
    # Apply the weights_init() function to randomly initialize all
    # weights to mean=0.0, stddev=0.2
    netD.apply(weights_init)
    # Print the model.
    print(netD)

    # Binary Cross Entropy loss function.
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, params["nz"], 1, 1, device=device)

    real_label = 1
    fake_label = 0

    # Optimizer for the discriminator.
    optimizerD = optim.Adam(
        netD.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )
    # Optimizer for the generator.
    optimizerG = optim.Adam(
        netG.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
    )

    # Stores generated images as training progresses.
    img_list = []
    # Stores generator losses during training.
    G_losses = []
    # Stores discriminator losses during training.
    D_losses = []

    iters = 0

    print("Starting Training Loop...")
    print("-" * 25)

    for epoch in range(params["nepochs"]):
        dataloader, test_data_loader = get_loaders(
            "./datasets/data", 32, 64, device, seq_len=params["nc"]
        )
        for i, (data, y) in enumerate(dataloader):
            y = y.squeeze(2)
            data = data.squeeze(2)
            # ipdb.set_trace()
            # Transfer data tensor to GPU/CPU (device)
            real_data = data
            # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
            b_size = real_data.size(0)

            # Make accumalated gradients of the discriminator zero.
            netD.zero_grad()
            # Create labels for the real data. (label=1)
            label = torch.full((b_size,), real_label, device=device).float()
            output = netD(y).view(-1)
            errD_real = criterion(output, label)
            # Calculate gradients for backpropagation.
            errD_real.backward()
            D_x = output.mean().item()

            # Sample random data from a unit normal distribution.
            # noise = torch.randn(b_size, params["nz"], 1, 1, device=device)
            # Generate fake data (images).
            fake_data = netG(data)
            # Create labels for fake data. (label=0)
            label.fill_(fake_label)
            # Calculate the output of the discriminator of the fake data.
            # As no gradients w.r.t. the generator parameters are to be
            # calculated, detach() is used. Hence, only gradients w.r.t. the
            # discriminator parameters will be calculated.
            # This is done because the loss functions for the discriminator
            # and the generator are slightly different.
            output = netD(fake_data.detach()).view(-1)
            errD_fake = criterion(output, label)
            # Calculate gradients for backpropagation.
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Net discriminator loss.
            errD = errD_real + errD_fake
            # Update discriminator parameters.
            optimizerD.step()

            # Make accumalted gradients of the generator zero.
            netG.zero_grad()
            # We want the fake data to be classified as real. Hence
            # real_label are used. (label=1)
            label.fill_(real_label)
            # No detach() is used here as we want to calculate the gradients w.r.t.
            # the generator this time.
            output = netD(fake_data).view(-1)
            errG = criterion(output, label)
            # Gradients for backpropagation are calculated.
            # Gradients w.r.t. both the generator and the discriminator
            # parameters are calculated, however, the generator's optimizer
            # will only update the parameters of the generator. The discriminator
            # gradients will be set to zero in the next iteration by netD.zero_grad()
            errG.backward()

            D_G_z2 = output.mean().item()
            # Update generator parameters.
            optimizerG.step()

            # Check progress of training.
            if i % 50 == 0:
                visualize_predictions(data, y, fake_data)
                print(
                    f"[{epoch}/{params['nepochs']}]\t"
                    + f"Loss_D: {errD.item():.4f}\t"
                    + f"Loss_G: {errG.item():.4f}\t"
                    + f"D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.14f} / {D_G_z2:.4f}"
                )

            # Save the losses for plotting.
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on a fixed noise.
            if iters % 100 == 0:
                with torch.no_grad():
                    fake_data = netG(data).detach().cpu()
            iters += 1

        # Save the model.
        if epoch % params["save_epoch"] == 0:
            torch.save(
                {
                    "generator": netG.state_dict(),
                    "discriminator": netD.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                    "params": params,
                },
                os.path.join(curdir, "model/model_epoch_{}.pth".format(epoch)),
            )

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
