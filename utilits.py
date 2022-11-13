from torch.utils.data import DataLoader, Subset, random_split, Dataset
import pandas as pd
import numpy as np
import torch as th


class MyDataset(Dataset):
    def __init__(self, data, label, **hyperparameters):
        super(MyDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.label[i]