import h5py
import torch as t


class DataManger:
    def __init__(self, *, data_path: str = "", ranges: t.Tensor = None):
        # read data_path in h5 format
        if data_path is not None:
            self.ranges = t.from_numpy(h5py.File(data_path, "r")["ranges"])
        elif ranges is not None:
            self.ranges = ranges
        else:
            raise ValueError("data_path or ranges must be specified")

    def normalize(self, data: t.Tensor):
        max_val = self.ranges[:, 0]
        min_val = self.ranges[:, -1]
        return 0.1 + 0.9 * (data - min_val) / (max_val - min_val)

    def denormalize(self, data: t.Tensor):
        max_val = self.ranges[:, 0]
        min_val = self.ranges[:, -1]
        return ((data - 0.1) * (max_val - min_val)) / 0.9 - max_val

    def discretize(self, data):
        threshold = self.ranges[:, 1]
        return data > threshold
