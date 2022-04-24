from argparse import ArgumentParser
import os
import torch as t
import json
import ipdb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from .data_loader import DeepCoastalDataModule
from .gan_logger import GANLogger

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
        "action", help="train or test the model", choices=["train", "test", "merge-results"] # TODO: add merge-results
    )
    for key, value in default_params.items():
        parser.add_argument(
            "--" + key, help=key + " to use", default=value, type=type(value),
        )
    params = parser.parse_args()
    training_dict = {
        key: val for key, val in params.__dict__.items() if key != "action"
    }
    print(json.dumps(training_dict, indent=4))
    cur_folder = os.path.basename(os.path.dirname(__file__))
    module = __import__(
        f"{cur_folder}.models.{params.model}.model", fromlist=["models"]
    )
    model = module.Model(params)
    save_path = os.path.join(cur_folder, "models", params.model)
    params.save_path = save_path
    print("Training")
    with open(os.path.join(save_path, "train_params.json"), "w") as f:
        json.dump(training_dict, f, indent=4)
    trainer = Trainer(
        max_epochs=params.max_epochs,
        gpus=(1 if params.cuda else None),
        callbacks=[
            EarlyStopping(
                monitor="val_mse", patience=params.early_stopping_patience
            )
        ],
        logger=GANLogger(params),
    )
    data_module = DeepCoastalDataModule(params)
    model_path = os.path.join(save_path, "model.pt")
    if params.action == "train":
        trainer.fit(model=model, datamodule=data_module)
        t.save(model.state_dict(), model_path)

    if os.path.exists(model_path):
        print(f"\nLoading model from '{model_path}'\n")
        model.load_state_dict(t.load(model_path))
    trainer.test(model=model, datamodule=data_module)
     
if __name__ == "__main__":
    run()
