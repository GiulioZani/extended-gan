import torch as t
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import ipdb


class precipitation_maps_h5(Dataset):
    def __init__(
        self,
        in_file,
        num_input_images,
        num_output_images,
        train=True,
        transform=None,
    ):
        super(precipitation_maps_h5, self).__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.sequence_length = num_input_images + num_output_images

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images - (
            num_input_images + num_output_images
        )
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_name, "r", rdcc_nbytes=1024 ** 3
            )["train" if self.train else "test"]["images"]
        imgs = np.array(
            self.dataset[index : index + self.sequence_length], dtype="float32"
        )

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        return self.size_dataset


class precipitation_maps_oversampled_h5(Dataset):
    def __init__(
        self,
        in_file: str,
        num_input_images: int,
        num_output_images: int,
        train: bool = True,
        transform=None,
    ):
        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, "r")[
            "train" if train else "test"
        ]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset = h5py.File(
                self.file_name, "r", rdcc_nbytes=1024 ** 3
            )["train" if self.train else "test"]["images"]
        imgs = np.array(self.dataset[index], dtype="float32")
        # print("******SHAPE: ", imgs.shape)
        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        # target_img = imgs[-1] # Modified
        target_img = imgs[self.num_input :]

        input_img = t.from_numpy(input_img).permute(2, 3, 1, 0)
        target_img = t.from_numpy(target_img).permute(2, 3, 1, 0)
        # print(f"{input_img.shape} {target_img.shape}")

        return input_img, target_img

    def __len__(self):
        return self.samples


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.in_seq_len = params.in_seq_len
        self.out_seq_len = params.out_seq_len
        self.num_workers = os.cpu_count()

    def train_dataloader(self):
        return DataLoader(
            precipitation_maps_oversampled_h5(
                self.data_location, 6, 6, train=True
            )
        )

    def test_dataloader(self):
        return DataLoader(
            precipitation_maps_oversampled_h5(
                self.data_location, 6, 6, train=False
            )
        )

    def val_dataloader(self):
        return DataLoader(
            precipitation_maps_oversampled_h5(
                self.data_location, 6, 6, train=False
            )
        )
