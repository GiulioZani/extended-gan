from argparse import ArgumentParser
import os
from .test import test
from torch import optim
import torch as t
import json
import ipdb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from .data_loader import DeepCoastalDataModule

# in the main function if parses the arguments and initializes the appropiate models
def run():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, "default_parameters.json")) as f:
        default_params = json.load(f)

    parser = ArgumentParser()
    cur_folder = os.path.dirname(__file__)
    choices = tuple(
        x for x in os.listdir(os.path.join(cur_folder, "models")) if not "__" in x
    )
    parser.add_argument("model", help="model to use", choices=choices)
    parser.add_argument(
        "action", help="train or test the model", choices=["train", "test"]
    )
    for key, value in default_params.items():
        parser.add_argument(
            "--" + key, help=key + " to use", default=value, type=type(value),
        )
    params = parser.parse_args()
    print(json.dumps(params.__dict__, indent=4))
    cur_folder = os.path.basename(os.path.dirname(__file__))
    module = __import__(
        f"{cur_folder}.models.{params.model}.model", fromlist=["models"]
    )
    model = module.Model(params)
    if params.action == "train":
        print("Training")
        trainer = Trainer(
            max_epochs=params.max_epochs,
            gpus=(1 if params.cuda else None),
            callbacks=[
                EarlyStopping(
                    monitor="val_loss", patience=params.early_stopping_patience
                )
            ],
        )
        data_module = DeepCoastalDataModule(params)
        trainer.fit(model=model, datamodule=data_module)
        ipdb.set_trace()
    if params.action == "test":
        pass
        # test(
        #    params=params, temp_disc=temp_disc, frame_disc=frame_disc, gen=gen
        # )


if __name__ == "__main__":
    run()
