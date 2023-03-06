import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
 
class XLSParser():
    """ Load any XLS OR CSV file into a numpy array."""
    def __init__(self, xls_path) -> None:
        self.xls_path = xls_path
        if self.xls_path.endswith(".csv"):
            self.data = pd.read_csv(self.xls_path)
        else:
            self.data = pd.read_excel(self.xls_path)

    def parse(self, x_col : list, y_col : list) -> Tuple[np.ndarray, np.ndarray]:
        """ Parse the data into x and y values. """
        if x_col is None and y_col is None:
            self.x = self.data.iloc[:, :-1].values
            self.y = self.data.iloc[:, -1].values
        elif x_col is None:
            self.y = self.data.iloc[:, y_col].values
            self.x = self.data.drop(self.data.columns[y_col], axis=1).values
        elif y_col is None:
            self.x = self.data.iloc[:, x_col].values
            self.y = self.data.drop(self.data.columns[x_col], axis=1).values
        else:
            self.x = self.data.iloc[:, x_col].values
            self.y = self.data.iloc[:, y_col].values

        # assure that the data is in the right shape
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(-1, 1)
        if len(self.y.shape) == 1:
            self.y = self.y.reshape(-1, 1)

        return self.x, self.y
        
class XLSDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.array(x, dtype = np.float32)
        self.y = np.array(y, dtype = np.float32)
        
    def __len__(self):
        """
        Return the shape of the dataset. In this case, it is the number of data points.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Transform the data to torch.Tensor
        """
        return self.x[idx], self.y[idx]


def create_xls_dataloader(**kwargs):
    """
    A simple dataloader for the toy dataset.
    Args:
        batch_size (int): batch size
        test_ratio (float): ratio of test data
    
    Returns:
        train_loader (torch.utils.data.DataLoader): dataloader for training
        test_loader (torch.utils.data.DataLoader): dataloader for testing
    """
    xls_path = kwargs.pop("path")
    batch_size = kwargs.pop("batch_size", 32)
    cv_split_num = kwargs.pop("cv_split_num", 1)
    test_ratio = kwargs.pop("test_ratio", 0.2)

    parser = XLSParser(xls_path)

    x_col = kwargs.pop("x_col", None)
    y_col = kwargs.pop("y_col", None)

    x, y = parser.parse(x_col, y_col)
    num_test = int(x.shape[0] * test_ratio)

    if batch_size == -1:
        batch_size = x.shape[0]
        
    # create train and test data for cross validation
    for i in range(cv_split_num):
        x, y = shuffle(x, y)
        test_x,test_y = x[:num_test], y[:num_test]
        train_x, train_y = x[num_test:], y[num_test:]

        # create dataloader
        train_dataset = XLSDataset(train_x, train_y)
        test_dataset = XLSDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        yield train_loader, test_loader, train_dataset, test_dataset

    