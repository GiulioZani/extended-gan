import torch as t
from pytorch_lightning import LightningDataModule
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import ipdb


class CustomDataset(Dataset):
    def __init__(
        self,
        in_file: str,
        num_input_images: int,
        num_output_images: int,
        train: bool = True,
        transform=None,
        crop: int = 40,
    ):
        self.num_input = num_input_images
        self.num_output = num_output_images
        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["images"].shape

        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_name, "r", rdcc_nbytes=1024 ** 3
            )["train" if self.train else "test"]["images"]
        imgs = np.array(self.dataset[index], dtype="float32")
        print(f"{imgs.shape=}")
        # print("******SHAPE: ", imgs.shape)
        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        target_img = imgs[12:]
        # target_img = imgs[-1] # Modified
        input_img = imgs[6:12]

        input_img = t.from_numpy(input_img)[:, None]  # .permute(2, 3, 1, 0)
        target_img = t.from_numpy(target_img)[:, None]  # .permute(2, 3, 1, 0)
        print(f"{input_img.shape=} {target_img.shape=}")

        return input_img, target_img

    def __len__(self):
        return self.samples


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.in_seq_len = params.in_seq_len
        self.out_seq_len = params.out_seq_len
        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(
            CustomDataset(
                self.data_location,
                self.in_seq_len,
                self.out_seq_len,
                train=True,
            )
        )

    def test_dataloader(self):
        return DataLoader(
            CustomDataset(
                self.data_location,
                self.in_seq_len,
                self.out_seq_len,
                train=False,
            )
        )

    def val_dataloader(self):
        return DataLoader(
            CustomDataset(
                self.data_location,
                self.in_seq_len,
                self.out_seq_len,
                train=False,
            )
        )
