# CSV dataloader

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.utils import shuffle
from typing import *
import numpy as np


class CSVParser():
    """ Load any XLS file into a numpy array."""
    def __init__(self, csv_path) -> None:
        self.csv_path = csv_path
        self.data = pd.read_csv(self.csv_path)

    def parse(self, x_col : list, y_col : list) -> Tuple[np.ndarray, np.ndarray]:
        """ Parse the data into x and y values. """
        if x_col is None:
            self.x = self.data.iloc[:, :-1].values
        else:
            self.x = self.data.iloc[:, x_col].values
        
        if y_col is None:
            self.y = self.data.iloc[:, -1].values
        else:
            self.y = self.data.iloc[:, y_col].values

        # assure that the data is in the right shape
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(-1, 1)
        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)

        return self.x, self.y
        

class CSVDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        """
        Return the shape of the dataset. In this case, it is the number of data points.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Transform the data to torch.Tensor
        """
        return torch.Tensor(self.x[idx]), torch.Tensor(self.y[idx])


class CSVDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        """
        Return the shape of the dataset. In this case, it is the number of data points.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Transform the data to torch.Tensor
        """
        return torch.Tensor(self.x[idx]), torch.Tensor(self.y[idx])


def create_csv_dataloader(**kwargs):
    """
    A simple dataloader for the toy dataset.
    Args:
        batch_size (int): batch size
        test_ratio (float): ratio of test data
    
    Returns:
        train_loader (torch.utils.data.DataLoader): dataloader for training
        test_loader (torch.utils.data.DataLoader): dataloader for testing
    """
    csv_path = kwargs.pop("path")
    test_ratio = kwargs.pop("test_ratio", 0.2)
    batch_size = kwargs.pop("batch_size", 32)

    print()
    parser = CSVParser(csv_path)
    x,y = parser.parse(None,None)
    # create train and test data
    x, y = shuffle(x, y)

    # normalize x and y values
    x_mean,x_std = x.mean(axis=0), x.std(axis=0)
    y_mean,y_std = y.mean(axis=0), y.std(axis=0)

    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)

    num_test = int(x.shape[0] * test_ratio)
    train_x, train_y = x[num_test:], y[num_test:]
    test_x, test_y = x[:num_test], y[:num_test]

    # create dataloader
    train_dataset = CSVDataset(train_x, train_y)
    test_dataset = CSVDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset, (x_mean, x_std), (y_mean, y_std)

    
