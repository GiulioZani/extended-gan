import torch as t
import os
import h5py
import ipdb
from tqdm import tqdm
import netCDF4
from ..utils.data_manager import DataManger


def preprocess(in_dir: str = "/mnt/tmp/", out_file_name: str = "/mnt/tmp/data.h5"):
    vars = [
        ("Sea Surface Height", "SSH.nc", "zos"),
        ("Eastword Wind Speed", "CUR_uo.nc", "uo"),
        ("Northword Wind Speed", "CUR_vo.nc", "vo"),
        # ("Temperature", "TEM.nc", "thetao"),
        ("Salinity", "SAL.nc", "so"),
    ]
    # fig, axes = plt.subplots(ncols=len(vars), nrows=1)
    acc = []
    ranges = []
    for (simple_name, file_name, var_name) in tqdm(vars):
        a = netCDF4.Dataset(os.path.join(in_dir, file_name))
        raw_data = t.from_numpy(a[var_name][...]).squeeze()
        subsampled_data = raw_data[:, :, :65]
        # raw_data = raw_data[20:, :65]
        # ax.imshow(raw_data.squeeze())
        # ax.title.set_text(simple_name)
        # ax.axis("off")
        # plt.savefig(f"{simple_name}")
        second_min = t.min(subsampled_data[subsampled_data != t.min(subsampled_data)])
        subsampled_data[subsampled_data == t.min(subsampled_data)] = second_min
        ranges.append(
            (
                t.max(subsampled_data).item(),
                t.mean(subsampled_data).item(),
                t.min(subsampled_data).item(),
            )
        )

        acc.append(subsampled_data)
    ranges_tensor = t.tensor(ranges)
    # min_len = min([a.shape[0] for a in acc])
    # acc = [a[:min_len] for a in acc]
    # saves file as hdf5
    data_manager = DataManger(ranges=ranges_tensor)
    print("saving..")
    result = data_manager.normalize(t.stack(acc, dim=1))
    with h5py.File(out_file_name, "w") as f:
        f.create_dataset(f"default", data=result)
    return ranges_tensor
    # t.save(result, "coastal_sea_data_preprocessed.pt")
    # plt.savefig("variables.png", dpi=200)


def train_test_split(
    ranges: t.Tensor,
    in_file_name="/mnt/tmp/data.h5",
    out_file: str = ".../datasets/4ch_coastal_normalization.h5",
):
    with h5py.File(in_file_name, "r") as f:
        data = f["default"][...]
    test_size = int(0.2 * len(data))
    test = data[-test_size:]
    train = data[:test_size]

    # saves train and test in h5py format
    with h5py.File(out_file, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("rages", data=ranges)

    """
    t.save(test_1, os.path.join(out_dir, "test", "1.pt"))
    t.save(test_2, os.path.join(out_dir, "test", "2.pt"))
    t.save(train, os.path.join(out_dir, "train", "1.pt"))
    """


if __name__ == "__main__":
    intermediate = "/mnt/tmp/data.h5"
    ranges = preprocess("/mnt/tmp/", intermediate)
    train_test_split(ranges, intermediate)
