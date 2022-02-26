import os
import torch as t
import ipdb
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(
        self,
        folder: str,
        batch_size: int,
        device: t.device,
        *,
        crop=64,
        shuffle: bool = True,
        seq_len: int = 4
    ):
        self.seq_len = seq_len
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
        data = data[: (len(data) // 2 * self.seq_len) * 2 * self.seq_len]
        if self.crop is not None:
            data = data[:, :, : self.crop, : self.crop]

        segments = t.stack(
            tuple(
                el
                for el in tuple(
                    data[i : i + 2 * self.seq_len] for i in range(len(data))
                )
                if len(el) == 2 * self.seq_len
            )
        )
        return segments

    def __next__(self) -> tuple[t.Tensor, t.Tensor]:
        if self.remainder.shape[1] == 0:
            data = self.__read_next_file()
        else:
            data = self.remainder
        self.remainder = data[self.batch_size :]
        result = data[: self.batch_size]
        if len(result) == 0:
            raise StopIteration
        result = t.stack(
            tuple(
                t.stack((s[: self.seq_len], s[self.seq_len :])) for s in result
            )
        ).transpose(0, 1)
        rand_indices = (
            t.randperm(result.shape[1])
            if self.shuffle
            else t.arange(result.shape[1])
        )
        results = (
            result[0][rand_indices].float().to(self.device),
            result[1][rand_indices].float().to(self.device),
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
    seq_len: int = 4
) -> tuple[DataLoader, DataLoader]:
    test_folder = os.path.join(data_location, "test")
    train_folder = os.path.join(data_location, "train")
    return (
        DataLoader(train_folder, train_batch_size, device, seq_len=seq_len),
        DataLoader(test_folder, test_batch_size, device, seq_len=seq_len),
    )


def test():
    train_dl, test_dl = get_loaders(
        "../datasets/data",
        32,
        64,
        t.device("cuda" if t.cuda.is_available() else "cpu"),
    )
    for (x, y) in test_dl:
        plt.imshow(x[0, 0, 0].cpu())
        plt.show()
        print(x.shape)
        return


if __name__ == "__main__":
    test()
