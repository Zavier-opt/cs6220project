from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from math import floor
from pytorch_lightning import LightningDataModule
# To Avoid Crashes with a lot of nodes
import torch.multiprocessing

from utilits import MovieLensDataset

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

        # # Singletons of MNIST train and test datasets
        # if MovieFederatedDM.mnist_train is None:
        #     MovieFederatedDM.mnist_train = MNIST("", train=True, download=True, transform=transforms.ToTensor())
        #     if not iid:
        #         sorted_indexes = MovieFederatedDM.mnist_train.targets.sort()[1]
        #         MovieFederatedDM.mnist_train.targets = MovieFederatedDM.mnist_train.targets[sorted_indexes]
        #         MovieFederatedDM.mnist_train.data = MovieFederatedDM.mnist_train.data[sorted_indexes]
        # if MovieFederatedDM.mnist_val is None:
        #     MovieFederatedDM.mnist_val = MNIST("", train=False, download=True, transform=transforms.ToTensor())
        #     if not iid:
        #         sorted_indexes = MovieFederatedDM.mnist_val.targets.sort()[1]
        #         MovieFederatedDM.mnist_val.targets = MovieFederatedDM.mnist_val.targets[sorted_indexes]
        #         MovieFederatedDM.mnist_val.data = MovieFederatedDM.mnist_val.data[sorted_indexes]
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

        trainset = MovieLensDataset(train_rating_path, train_usermovie_path, None, None)
        # rows_by_sub = floor(len(trainset) / self.number_sub)
        # tr_subset = Subset(trainset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub))
        movieuser_train, movieuser_val = random_split(trainset, [round(len(trainset) * (1 - self.val_percent)),
                                                                 round(len(trainset) * self.val_percent)])

        # Test set
        movieuser_test = MovieLensDataset("data/test/test_r.csv", "data/test/test_u.csv", None, None)
        # rows_by_sub = floor(len(testset) / self.number_sub)
        # te_subset = Subset(testset, range(self.sub_id * rows_by_sub, (self.sub_id + 1) * rows_by_sub))

        # if len(testset) < self.number_sub:
        #     raise ("Too much partitions")

        # DataLoaders
        self.train_loader = DataLoader(movieuser_train, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.val_loader = DataLoader(movieuser_val, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers)
        self.test_loader = DataLoader(movieuser_test, batch_size=self.batch_size, shuffle=False,
                                      num_workers=self.num_workers)
        print("Train: {} Val:{} Test:{}".format(len(movieuser_train), len(movieuser_val), len(movieuser_test)))

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
