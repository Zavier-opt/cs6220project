import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy


###############################
#    Multilayer Perceptron    #
###############################

class DNN(pl.LightningModule):
    """
    Multilayer Perceptron (MLP) to solve MNIST with PyTorch Lightning.
    """

    def __init__(self, metric=Accuracy, lr_rate=0.001, seed=None):  # low lr to avoid overfitting

        # Set seed for reproducibility iniciialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        super().__init__()
        n_users, n_movies = 943, 1664
        n_factors = 50
        embedding_dropout = 0.02
        hidden = 10
        dropouts = 0.2

        self.lr_rate = lr_rate
        self.metric = metric()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden = nn.Sequential(
            nn.Linear(n_factors * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropouts),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropouts),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropouts),
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, users, movies, minmax=None):
        """
        """
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.drop(features)
        x = self.hidden(x)
        out = self.fc(x)
        return out

    def configure_optimizers(self):
        """
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr_rate)

    def training_step(self, batch, batch_id):
        """
        """
        x, y = batch
        minmax = (1.0,5.0)
        criterion = nn.MSELoss()
        outputs = self(x[:, 0], x[:, 1], minmax=minmax)
        loss = criterion(outputs, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        minmax = (1.0,5.0)
        criterion = nn.MSELoss()
        outputs = self(x[:, 0], x[:, 1], minmax=minmax)
        loss = criterion(outputs, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        """
        x, y = batch
        minmax = (1.0,5.0)
        criterion = nn.MSELoss()
        outputs = self(x[:, 0], x[:, 1], minmax=minmax)
        loss = criterion(outputs, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss
