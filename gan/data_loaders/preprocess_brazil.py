# from argparse import Namespace
# import os
# from cv2 import threshold
# from sqlalchemy import false
# import torch as t
# import ipdb
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.utils.data import DataLoader, Dataset, random_split
# from pytorch_lightning import LightningDataModule
# from PIL import Image
# from torchvision.transforms import transforms


# class CustomDataModule:
#     def __init__(self, params):
#         super().__init__()

#         self.data_location = params.data_location

#         self.in_seq_len = params.in_seq_len
#         self.out_seq_len = params.out_seq_len

#         # num workers = number of cpus to use
#         # get number of cpu's on this device
#         num_workers = os.cpu_count()
#         self.num_workers = num_workers

#         self.data = self.load_pytorch_tensor(os.path.join(self.data_location))

#         # change data type to float
#         self.data = self.data.float()

#         # normalize data with min-max normalization and scale to -1 to 1
#         # self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

#         # segment data into sequences
#         self.data = self.segment_data(self.data)
#         self.data = self.threshold_data(self.data, 1000)



#     def analyze_data(self):
#         """
#         Plots the data.
#         """
#         # get histogram of rainfall distribution, sum over all segments

#         # get histogram of rainfall distribution, sum over all segments
#         # plt.hist(self.data.numpy().sum(axis=1), bins=100)
#         # ipdb.set_trace()
#         data = self.data.view(self.data.shape[0], -1)

#         data = data[data.sum(1) > 700]

#         plt.hist(data.numpy().sum(axis=1), bins=100)
#         plt.show()

#     def threshold_data(self, data, threshold):
#         """
#         Thresholds data.
#         """
#         flat_data = data.view(data.shape[0], -1)
#         return data[flat_data.sum(1) > threshold]

#     def segment_data(self, data):
#         """
#         Segments data into sequences of length in_seq_len + out.
#         """

#         # ipdb.set_trace()
#         tot_lenght = self.in_seq_len + self.out_seq_len
#         data = data[: (len(data) // tot_lenght) * tot_lenght]
#         # ipdb.set_trace()
#         segments = t.stack(
#             tuple(data[i : i + tot_lenght, ...] for i in range(len(data) - tot_lenght))
#         )
#         return segments

#     def load_pytorch_tensor(self, file_path):
#         """
#         Loads a PyTorch tensor from a file.
#         """
#         with open(file_path, "rb") as f:
#             return t.load(f)


# # if __name__ == "__main__":
#     # ipdb.set_trace()
#     # pass
#     # params = Namespace(
#     #     data_location="../datasets/brazil_data.pt",
#     #     in_seq_len=10,
#     #     out_seq_len=10,
#     # )
#     # data_module = CustomDataModule(params)
