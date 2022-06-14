import os
import torch as t
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from pytorch_lightning import LightningDataModule
from PIL import Image
from torchvision.transforms import transforms


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()

        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.in_seq_len = params.in_seq_len
        self.out_seq_len = params.out_seq_len

        self.data = self.load_pytorch_tensor(os.path.join(self.data_location))
        
        # change data type to float
        self.data = self.data.float()

        # normalize data with min-max normalization and scale to -1 to 1
        self.data = (
            2 * (self.data - self.data.min()) / (self.data.max() - self.data.min()) - 1
        )

        # self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

        # random split into train and test

        # random seed torch
        t.manual_seed(42)
        self.train_data, self.test_data = random_split(
            self.data,
            [len(self.data) - int(len(self.data) * 0.1), int(len(self.data) * 0.1)],
        )
        # ipdb.set_trace()
        # segment data into sequences
        self.train_data = self.segment_data(self.train_data.dataset)
        self.test_data = self.segment_data(self.test_data.dataset)
        self.train_data = CustomDataset(
            self.train_data, self.in_seq_len, self.out_seq_len
        )
        self.test_data = CustomDataset(
            self.test_data, self.in_seq_len, self.out_seq_len
        )

    def segment_data(self, data):
        """
        Segments data into sequences of length in_seq_len + out.
        """

        # ipdb.set_trace()
        tot_lenght = self.in_seq_len + self.out_seq_len
        data = data[: (len(data) // tot_lenght) * tot_lenght]
        # ipdb.set_trace()
        segments = t.stack(
            tuple(data[i : i + tot_lenght, ...] for i in range(len(data) - tot_lenght))
        )
        return segments

    def load_pytorch_tensor(self, file_path):
        """
        Loads a PyTorch tensor from a file.
        """
        with open(file_path, "rb") as f:
            return t.load(f)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.train_batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=True)


class CustomDataset(Dataset):
    def __init__(self, data, in_seq_len, out_seq_len):
        super().__init__()
        # ipdb.set_trace()
        self.data = data
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ipdb.set_trace()
        return self.data[idx, : self.in_seq_len].unsqueeze(1), self.data[idx, self.in_seq_len :].unsqueeze(1)
