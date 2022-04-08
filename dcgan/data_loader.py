import os
import torch as t
import ipdb
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataLoader:
    def __init__(
        self,
        folder: str,
        batch_size: int,
        device: t.device,
        *,
        crop=64,
        shuffle: bool = True,
        in_seq_len: int = 4,
        out_seq_len: int = 4
    ):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.tot_seq_len = in_seq_len + out_seq_len
        self.crop = crop
        self.data_folder = folder
        self.device = device
        self.batch_size = batch_size
        self.file_index = 0
        self.folder = folder
        self.files = tuple(
            os.path.join(folder, fn) for fn in sorted(os.listdir(folder))
        )
        self.shuffle = shuffle
        if self.shuffle:
            rand_indices = t.randperm(len(self.files))
            tmp = tuple(self.files[i] for i in rand_indices)
            self.files = tmp
        self.remainder = self.__read_next_file()
        self.file_length = self.remainder.shape[0] * self.remainder.shape[1]

    def __read_next_file(self) -> t.Tensor:
        if self.file_index == len(self.files):
            raise StopIteration
        data = t.load(self.files[self.file_index])
        self.file_index += 1
        result = self.__segmentify(data)
        return result

    def __segmentify(self, data: t.Tensor) -> t.Tensor:
        data = data[: (len(data) // self.tot_seq_len) * self.tot_seq_len]
        if self.crop is not None:
            data = data[:, :, : self.crop, : self.crop]
        return data 

    def __next__(self) -> tuple[t.Tensor, t.Tensor]:
        if self.remainder.shape[0] == 0:
            data = self.__read_next_file()
        else:
            data = self.remainder
        self.remainder = data[self.batch_size:]
        result = t.stack(tuple(data[i:i+self.tot_seq_len] for i in range(self.batch_size)))
        if len(result) == 0:
            raise StopIteration
        xs = t.stack(
            tuple(
                s[: self.in_seq_len]
                for s in result
            )
        )
        ys = t.stack(
            tuple(
                s[self.in_seq_len:]
                for s in result
            )
        )
        rand_indices = (
            t.randperm(result.shape[0])
            if self.shuffle
            else t.arange(result.shape[0])
        )
        results = (
            xs[rand_indices].float().to(self.device),
            ys[rand_indices].float().to(self.device),
        )
        return results

    def __iter__(self):
        return self


def get_loaders(
    data_location: str,
    train_batch_size: int,
    test_batch_size: int,
    device: t.device,
    *,
    crop: int = 64,
    in_seq_len: int = 12,
    out_seq_len: int = 6
) -> tuple[DataLoader, DataLoader]:
    test_folder = os.path.join(data_location, "test")
    train_folder = os.path.join(data_location, "train")
    return (
        DataLoader(
            train_folder,
            train_batch_size,
            device,
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            crop=64,
        ),
        DataLoader(
            test_folder,
            test_batch_size,
            device,
            in_seq_len=in_seq_len,
            out_seq_len=out_seq_len,
            crop=64,
        ),
    )


def test():
    train_dl, test_dl = get_loaders(
        "/mnt/tmp/multi_channel_train_test",
        32,
        64,
        t.device("cuda" if t.cuda.is_available() else "cpu"),
    )
    for i, (x, y) in enumerate(tqdm(train_dl)):
        # plt.imshow(x[0, 0, 0].cpu())
        # plt.show()
        # print(x.shape)
        # return
        print(f"iteration: {i}")


if __name__ == "__main__":
    test()
