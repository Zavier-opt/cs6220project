import pandas as pd
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from math import floor
from pytorch_lightning import LightningDataModule
# To Avoid Crashes with a lot of nodes
import torch.multiprocessing

from utilits import MyDataset

torch.multiprocessing.set_sharing_strategy('file_system')


#######################################
#    FederatedDataModule for MNIST    #
#######################################

class MovieFederatedDM(LightningDataModule):
    """
    LightningDataModule of partitioned MNIST. Its used to generate **IID** distribucions over MNIS. Toy Problem.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    # Singleton
    mnist_train = None
    mnist_val = None

    def __init__(self, experiment="user", num_of_split=1, sub_id=0, number_sub=1, batch_size=32, num_workers=4,
                 val_percent=0.1):
        super().__init__()
        self.experiment = experiment
        self.number_of_split = num_of_split
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent

        if self.sub_id + 1 > self.number_sub:
            raise ("Not exist the subset {}".format(self.sub_id))
        if self.experiment in {"user", "movie"}:
            exp = self.experiment
        else:
            raise "The input experiment is invalid"
        if self.number_of_split in {1, 2, 3, 4}:
            num_of_split = self.number_of_split
        else:
            raise "Invalid number of split "

        train_rating_path = "data/" + exp + "/rating" + str(num_of_split) + "_" + str(sub_id) + ".csv"
        train_usermovie_path = "data/" + exp + "/user" + str(num_of_split) + "_" + str(sub_id) + ".csv"

        # Training / validation set
        # Test set

        X_train = pd.read_csv(train_usermovie_path)[["user", "movie"]].to_numpy()
        y_train = pd.read_csv(train_rating_path)[["rating"]].to_numpy()

        X_train = torch.as_tensor(X_train)
        y_train = torch.as_tensor(y_train, dtype=torch.float).squeeze()

        X_test = pd.read_csv("data/test/test_u.csv")[["user", "movie"]].to_numpy()
        y_test = pd.read_csv("data/test/test_r.csv")[["rating"]].to_numpy()

        X_test = torch.as_tensor(X_test)
        y_test = torch.as_tensor(y_test, dtype=torch.float).squeeze()

        max_rating = 5.0
        min_rating = 1.0
        y_train = (y_train - min_rating) / (max_rating - min_rating)
        y_test = (y_test - min_rating) / (max_rating - min_rating)

        train_dataset = MyDataset(X_train, y_train)
        test_dataset = MyDataset(X_test, y_test)

        number_of_train = round(len(train_dataset) * (1 - self.val_percent))
        train_dataset, val_dataset = random_split(train_dataset, [number_of_train,len(train_dataset)-number_of_train])

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print("Train: {} Val:{} Test:{}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

    def train_dataloader(self):
        """
        """
        return self.train_loader

    def val_dataloader(self):
        """
        """
        return self.val_loader

    def test_dataloader(self):
        """
        """
        return self.test_loader
