import torch as t
import os
import h5py
import ipdb


def main(
    in_file_name="/mnt/tmp/data.hdf5",
    out_dir: str = "/mnt/tmp/multi_channel_train_test",
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(os.path.join(out_dir, "test")):
        os.mkdir(os.path.join(out_dir, "test"))
    if not os.path.exists(os.path.join(out_dir, "train")):
        os.mkdir(os.path.join(out_dir, "train"))

    # reads the data in h5 format
    with h5py.File(
        in_file_name, "r"
    ) as f:  # "r" means that hdf5 file is open in read-only mode
        data = t.from_numpy(f["default"][:])
    test_size = int(0.2 * len(data))
    to_cut = (test_size + 16) // 2
    test_1 = data[:to_cut]
    test_2 = data[-to_cut:]

    train = data[to_cut:-to_cut]

    assert len(train) + len(test_1) + len(test_2) == len(data), "whoops"
    # saves the data in h5 format
    with h5py.File(os.path.join(out_dir, "test", "test_1.h5"), "w") as f:
        f["default"] = test_1.numpy()
    with h5py.File(os.path.join(out_dir, "test", "test_2.h5"), "w") as f:
        f["default"] = test_2.numpy()
    with h5py.File(os.path.join(out_dir, "train", "train.h5"), "w") as f:
        f["default"] = train.numpy()
    """
    t.save(test_1, os.path.join(out_dir, "test", "1.pt"))
    t.save(test_2, os.path.join(out_dir, "test", "2.pt"))
    t.save(train, os.path.join(out_dir, "train", "1.pt"))
    """


if __name__ == "__main__":
    main()
