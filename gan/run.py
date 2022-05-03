from argparse import ArgumentParser
import os
import torch as t
import json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# TODO: add create
# TODO: add clone
# TODO: add merge-results
# TODO: fix restart
# TODO: add freeze feature
# TODO: add snapshot

def run():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, "default_parameters.json")) as f:
        default_params = json.load(f)

    parser = ArgumentParser()
    cur_folder = os.path.dirname(__file__)
    choices = tuple(
        x
        for x in os.listdir(os.path.join(cur_folder, "models"))
        if not "__" in x
    )
    parser.add_argument("model", help="model to use", choices=choices)
    parser.add_argument(
        "action",
        help="train or test the model",
        choices=(
            "train",
            "test",
            "restart",
            "merge-results",
        ),  # TODO: add merge-results
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
    model_module = __import__(
        f"{cur_folder}.models.{params.model}.model", fromlist=["models"]
    )
    data_loader_module = __import__(
        f"{cur_folder}.data_loader", fromlist=["data_loader"]
    )
    logger_module = __import__(f"{cur_folder}.logger", fromlist=["logger"])
    model = model_module.Model(params)
    save_path = os.path.join(cur_folder, "models", params.model)
    params.save_path = save_path
    print("Training")
    with open(os.path.join(save_path, "train_params.json"), "w") as f:
        json.dump(training_dict, f, indent=4)
    """
    import re

    checkpoint_files = sorted(
        tuple(
            (fn, int(re.findall(r"\d+", fn)[0]))
            for fn in os.listdir(save_path)
            if fn.endswith(".ckpt")
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    

    for fn, epoch in checkpoint_files[1:]:
        os.remove(os.path.join(save_path, fn))
    """
    """
    checkpoint_path = (
        os.path.join(save_path, checkpoint_files[0][0])
        if len(checkpoint_files) > 0
        else None
    )
    checkpoint_callback = ModelCheckpoint(dirpath=save_path)
    """
    checkpoint_path = os.path.join(save_path, "checkpoint.ckpt")
    if params.action == "train" and os.path.exists(checkpoint_path):
        action = "go"
        while action not in ("y", "n"):
            action = input(
                "Checkpiont file exists. Re-training will erase it. Continue? [y/n]\n"
            )
        if action == "y":
            os.remove(checkpoint_path)
        else:
            print("Ok, exiting.")
            return
    print(model.generator)
    trainer = Trainer(
        max_epochs=params.max_epochs,
        gpus=(1 if params.cuda else None),
        callbacks=[
            EarlyStopping(
                monitor="val_mse", patience=params.early_stopping_patience
            ),
            # checkpoint_callback,
        ],
        logger=logger_module.CustomLogger(params),
        enable_checkpointing=False
    )
    data_module = data_loader_module.CustomDataModule(params)
    if params.action in ("restart", "test"):
        if os.path.exists(checkpoint_path):
            print(f"\nLoading model from checkpoint:'{checkpoint_path}'\n")
            model.load_state_dict(t.load(checkpoint_path))
        else:
            raise Exception(
                "No checkpoint found. You must train the model first!"
            )
    if params.action in ("train", "restart"):
        trainer.fit(model=model, datamodule=data_module)
        print(f"Saving model to {checkpoint_path}")
        t.save(model.state_dict(), checkpoint_path)
    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    run()
