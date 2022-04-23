import torch as t
import os
import h5py
import ipdb
from tqdm import tqdm
import netCDF4


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
        normalized_data = 0.1 + 0.9 * (subsampled_data - t.min(subsampled_data)) / (
            t.max(subsampled_data) - t.min(subsampled_data)
        )
        acc.append(normalized_data)
    # min_len = min([a.shape[0] for a in acc])
    # acc = [a[:min_len] for a in acc]
    # saves file as hdf5
    print("saving..")
    result = t.stack(acc, dim=1)
    with h5py.File(out_file_name, "w") as f:
        f.create_dataset(f"default", data=result)
    # t.save(result, "coastal_sea_data_preprocessed.pt")
    # plt.savefig("variables.png", dpi=200)


def train_test_split(
    in_file_name="/mnt/tmp/data.h5", out_file: str = "4ch_coastal_normalization.h5"
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

    """
    t.save(test_1, os.path.join(out_dir, "test", "1.pt"))
    t.save(test_2, os.path.join(out_dir, "test", "2.pt"))
    t.save(train, os.path.join(out_dir, "train", "1.pt"))
    """


if __name__ == "__main__":
    intermediate = "/mnt/tmp/data.h5"
    preprocess("/mnt/tmp/", intermediate)
    train_test_split(intermediate)
