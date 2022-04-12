from argparse import ArgumentParser
import os
from .train import train
from torch import optim


# in the main function if parses the arguments and initializes the appropiate models
def main():
    parser = ArgumentParser()
    choices = os.listdir('models')
    parser.add_argument('model', help='model to use', choices=choices)
    parser.add_argument('action', help='train the model', choices=['train', 'test'])
    args = parser.parse_args()
    # dynamically import modeule named after the model
    module = __import__(f'models.{args.model}', fromlist=['models'])

    temp_disc = module.TemporalDiscriminator()
    frame_disc = module.FrameDiscriminator()
    gen = module.Generator()
    if args.action == 'train':
        args.__dict__.update({
            "bsize": 32,  # Batch size during training.
            "imsize": 64,  # Spatial size of training images. All images will be resized to this size during preprocessing.
            "nc": 6,  # Number of sequences to predic
            "nz": 64,  # Size of the Z latent vector (the input to the generator).
            "ngf": 64,  # Size of feature maps in the generator. The depth will be multiples of this.
            "ndf": 32,  # Size of features maps in the discriminator. The depth will be multiples of this.
            "nepochs": 10,  # Number of training epochs.
            "lr": 0.0002,  # Learning rate for optimizers
            "beta1": 0.5,  # Beta1 hyperparam for Adam optimizer
            "save_epoch": 2,
        })
        temp_disc_optim = optim.Adam(
            temp_disc.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
        )
        frame_disc_optim = optim.Adam(
            frame_disc.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
        )
        # Optimizer for the generator.
        gen_optim = optim.Adam(
            gen.parameters(), lr=params["lr"], betas=(params["beta1"], 0.999)
        )
        # train the model
        train(params=params, temp_disc, frame_disc, gen, gen_optim, )

if __name__ == '__main__':
    main()
