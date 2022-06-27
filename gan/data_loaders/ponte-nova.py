import os
import torch as t
import ipdb
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms


class CustomDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()

        self.data_location = params.data_location
        self.train_batch_size = params.train_batch_size
        self.test_batch_size = params.test_batch_size
        self.in_seq_len = params.in_seq_len
        self.out_seq_len = params.out_seq_len
        self.tot_seq_len = params.in_seq_len + params.out_seq_len

        # num workers = number of cpus to use
        # get number of cpu's on this device
        num_workers = os.cpu_count()
        self.num_workers = num_workers

        data = self.load_pytorch_tensor(os.path.join(self.data_location))

        # change data type to float
        data = data.float()

        print("Data shape: {}".format(data.shape))

        t.manual_seed(42)
        # random split into train and test
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]

        # overlapping windows of length seq_size with sliding window of windows size
        self.train_data = self.segment_and_threshold_data(
            self.train_data, params.data_threshold, params.sliding_window_size
        )

        # non overlapping segments applied with threshold
        self.test_data = t.stack(
            [
                self.test_data[i : i + self.tot_seq_len]
                for i in range(
                    0, len(self.test_data) - self.tot_seq_len, self.tot_seq_len
                )
            ]
        )

        print("Train Data Size: {}".format(len(self.train_data)))

        self.transform = None

        # min max normalize and map to -1 to 1

        max = t.max(data)
        min = t.min(data)

        self.train_data = 2 * (self.train_data - min) / (max - min) - 1
        self.test_data = 2 * (self.test_data - min) / (max - min) - 1

        self.train_data = Wrapper(
            self.train_data,
            self.in_seq_len,
            self.out_seq_len,
            transforms=self.transform,
        )
        self.test_data = Wrapper(
            self.test_data, self.in_seq_len, self.out_seq_len, transforms=self.transform
        )

    def threshold_data(self, data, threshold):
        """
        Thresholds data.
        """
        flat_data = data.view(data.shape[0], -1)
        return data[flat_data.sum(1) > threshold]

    def segment_and_threshold_data(self, data, threshold, sliding_window_size=1):

        # iterate over data sequences, only accept sequences of length in_seq_len + out_seq_len where sum of data is greater than threshold

        seq_size = self.in_seq_len + self.out_seq_len
        data_seq_size = data.shape[0]
        # ipdb.set_trace()

        # overlaping windows of length seq_size with sliding window of 1
        segments = [
            data[i : i + seq_size]
            for i in range(0, data_seq_size - seq_size, sliding_window_size)
            if t.sum(data[i : i + seq_size]) > threshold
        ]

        return t.stack(segments)

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
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class Wrapper(Dataset):
    def __init__(self, data, in_seq_len, out_seq_len, transforms=None):
        super().__init__()
        self.data = data
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.transforms is not None:
            data = self.transforms(self.data[idx])
        else:
            data = self.data[idx]

        return data[: self.in_seq_len].unsqueeze(1), data[self.in_seq_len :].unsqueeze(
            1
        )


class SubsetWrapper(Subset):
    def __init__(self, super_subset: Subset, in_seq_len, out_seq_len) -> None:
        super().__init__(super_subset.dataset, super_subset.indices)
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if isinstance(item, list):
            return [
                item[0][: self.in_seq_len].unsqueeze(1),
                item[1][: self.out_seq_len].unsqueeze(1),
            ]
        return item[: self.in_seq_len].unsqueeze(1), item[self.in_seq_len :].unsqueeze(
            1
        )
