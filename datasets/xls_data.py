import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import pandas as pd
 
class XLSParser():
    """
    Parser for the `.csv`or `.xls` tabular data files. The data will be parsed into x and y values.
    """
    def __init__(self, xls_path : Union[str,os.PathLike]) -> None:
        """
        xls_path (Union[str, os.PathLike]) : Path to the xls file.
        """
        self.xls_path = xls_path
        if self.xls_path.endswith(".csv"):
            self.data = pd.read_csv(self.xls_path)
        else:
            self.data = pd.read_excel(self.xls_path)

    def parse(self, x_col : Optional[list], y_col : Optional[list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse the data into x and y values.

        Args:
            - x_col (Optional[list]) : Column index of the x values. If None, all columns except the last one will be used.
            - y_col (Optional[list]) : Column index of the y values. If None, the last column will be used.
        
        Returns:
            - x (np.ndarray) : x values.
            - y (np.ndarray) : y values.
        """
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
    def __init__(self, x : np.array, y : np.array, transforms : Tuple):
        self.x = np.array(x, dtype = np.float32)
        self.y = np.array(y, dtype = np.float32)

        self.x_transform, self.y_transform = transforms

    def __len__(self):
        """
        Return the shape of the dataset. In this case, it is the number of data points.
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        Transform the data to torch.Tensor
        """
        return self.x_transform(self.x[idx]), self.y_transform(self.y[idx])


def create_xls_dataloader(
        transforms : Tuple,
        xls_path : Union[str, os.PathLike],
        batch_size : int = 32,
        cv_split_num : int = 1,
        test_ratio : float = 0.2,
        x_col : Optional[list] = None,
        y_col : Optional[list] = None,
        **kwargs):
    """
    A simple dataloader for the reading XLS/CSV files. The data will be split into a training and validation set.
    
    Args:
        - transforms (Tuple[Transform,Transform]) : List of x and y transform that will be applied when sampled.
            Transforms are composed with `transforms.Transform` class. 
        - xls_path (Union[str, os.PathLike]) : Path to the xls file.
        - batch_size (int) : Batch size for the dataloader. `-1` denotes that the whole dataset will be used.
        - cv_split_num (int) : Number of cross validation splits.
        - test_ratio (float) : Ratio of the test set.
        - x_col (Optional[list]) : Column index of the x values. If None, all columns except the last one will be used.
        - y_col (Optional[list]) : Column index of the y values. If None, the last column will be used.

    Returns:
        - train_loader (DataLoader) : Dataloader for the training set.
        - val_loader (DataLoader) : Dataloader for the validation set.
        - train_dataset (XLSDataset) : Dataset for the training set.
        - val_dataset (XLSDataset) : Dataset for the validation set.
    """

    parser = XLSParser(xls_path)
    x, y = parser.parse(x_col, y_col)
    num_test = int(x.shape[0] * test_ratio)

    if batch_size == -1:
        batch_size = x.shape[0]
        
    # create train and test data for cross validation
    for _ in range(cv_split_num):
        x, y = shuffle(x, y)
        test_x,test_y = x[:num_test], y[:num_test]
        train_x, train_y = x[num_test:], y[num_test:]

        # create dataloader
        train_dataset = XLSDataset(train_x, train_y, transforms)
        val_dataset = XLSDataset(test_x, test_y, transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        yield train_loader, val_loader, train_dataset, val_dataset

    