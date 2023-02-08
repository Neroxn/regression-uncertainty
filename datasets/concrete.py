import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
 
class XLSParser():
    """ Load any XLS file into a numpy array."""
    def __init__(self, xls_path) -> None:
        self.xls_path = xls_path
        self.data = pd.read_excel(self.xls_path)


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
        

class XLSDataset(Dataset):
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


def create_xls_dataloader(batch_size, xls_path, test_ratio=0.2):
    """
    A simple dataloader for the toy dataset.
    Args:
        batch_size (int): batch size
        test_ratio (float): ratio of test data
    
    Returns:
        train_loader (torch.utils.data.DataLoader): dataloader for training
        test_loader (torch.utils.data.DataLoader): dataloader for testing
    """
    parser = XLSParser(xls_path)
    x, y = parser.parse(None, None)

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
    train_dataset = XLSDataset(train_x, train_y)
    test_dataset = XLSDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_dataset, test_dataset

if __name__ == "__main__":
    xls_path = "Concrete_Data.xls"
    train_loader, test_loader, train_ds, test_ds = create_xls_dataloader(2, xls_path)
    print(next(iter(train_loader)), )


    