from torch.utils.data import DataLoader, Subset, random_split, Dataset
import pandas as pd
import numpy as np
import torch as th


class MovieLensDataset(Dataset):
    def __init__(self, rating_file, UserMovie_file, transform=None, target_transform=None):
        """
        annotations_file: ratings file (ratings.csv looks like : id, score)
        """
        self.ratings = pd.read_csv(rating_file);
        self.UserMovie_file = UserMovie_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        self.UserMovie = pd.read_csv(self.UserMovie_file)
        usermovie = np.asarray([self.UserMovie.iloc[idx, 1], self.UserMovie.iloc[idx, 2]])
        label = self.ratings.iloc[idx, 1]
        usermovie = th.from_numpy(usermovie)
        if self.target_transform:
            label = self.target_transform(label)
        return usermovie, label
