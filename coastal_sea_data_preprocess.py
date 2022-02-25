import torch as t
import ipdb
import netCDF4
import matplotlib.pyplot as plt


def main():
    acc = []
    for var_name, file_name in [("uo", "/mnt/tmp/CUR_uo.nc")]:
        raw_data = t.from_numpy(netCDF4.Dataset(file_name)[var_name][...])
        subsampled_data = raw_data[:, 0, 20:, :65]
        normalized_data = (subsampled_data - t.min(subsampled_data)) / (
            t.max(subsampled_data) - t.min(subsampled_data)
        )
        acc.append(normalized_data)
    result = t.stack(acc, dim=1)
    t.save(result, "coastal_sea_data_preprocessed_cur.pt")


if __name__ == "__main__":
    main()
