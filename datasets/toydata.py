import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class ToyDataset(Dataset):
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

def make_toy_dataset(test_ratio = 0.2, func : Callable = None, data_range = 3, data_step = 0.005, data_sigma1 = 3, data_sigma2 = 1, num_data = 1):
    """
    Make toy dataset with respect to the DeepEnsamble paper.
    Args:
        test_ratio (float): ratio of test data to total data
    
    Returns:
        train_x (np.ndarray): training data
        train_y (np.ndarray): training labels
        test_x (np.ndarray): test data
        test_y (np.ndarray): test labels
    """
    data_x = np.arange(-data_range, data_range + data_step, data_step)
    data_x = np.reshape(data_x, [data_x.shape[0], 1])
    data_x_true = data_x

    data_y = np.zeros([data_x.shape[0], 1])
    data_y_true = np.zeros([data_x.shape[0], 1])

    for i in range(data_x.shape[0]):
        if (data_x[i,0] < 0): 
            data_y[i, 0] = 10 * func(data_x[i,0]) + np.random.normal(0, data_sigma1)
        else:
            data_y[i, 0] = 10 * func(data_x[i,0]) + np.random.normal(0, data_sigma2)
            
        data_y_true[i, 0] = 10 * func(data_x[i,0])

    data_x, data_y = shuffle(data_x, data_y)
    num_train_data = int(data_x.shape[0] * (1 - test_ratio))

    train_x = data_x[:num_train_data, :]
    train_y = data_y[:num_train_data, :]
    test_x  = data_x[num_train_data:, :]
    test_y  = data_y[num_train_data:, :]

    print("Train data shape: " + str(train_x.shape))
    print("Test data shape: " + str(test_x.shape))

    plt.plot(data_x, data_y, 'b*')
    plt.plot(data_x_true, data_y_true, 'r')
    plt.legend(['Data', 'y=x^3'], loc = 'best')
    plt.title('y = 10sin(x) + $\epsilon$ where $\epsilon$ ~ N(0, 3^2) and N(0, 1^2)')
    plt.show()

    return train_x, train_y, test_x, test_y

def make_toy_dataset2(test_ratio = 0.2, func : Callable = None):
    data_range = 7
    data_step = 0.001

    bound1 = -2
    bound2 = 2

    data_sigma1 = 0.1
    data_sigma2 = 0.5
    num_data = 1

    data_x1 = np.arange(-data_range, bound1 + data_step, data_step)
    data_x2 = np.arange(bound2, data_range + data_step, data_step)
    data_x = np.concatenate((data_x1, data_x2))
    data_x = np.reshape(data_x, [data_x.shape[0], 1])

    data_y = np.zeros([data_x.shape[0], 1])

    data_y1_true = func(data_x1)
    data_y2_true = func(data_x2)

    for i in range(data_x.shape[0]):
        if (data_x[i,0] < bound1): 
            data_y[i, 0] = func(data_x[i,0]) + np.random.normal(0, data_sigma1)
        else:
            data_y[i, 0] = func(data_x[i,0]) + np.random.normal(0, data_sigma2)

    data_x, data_y = shuffle(data_x, data_y)
            
    num_train_data = int(data_x.shape[0] * (1 - test_ratio))
    num_test_data  = data_x.shape[0] - num_train_data

    train_x = data_x[:num_train_data, :]
    train_y = data_y[:num_train_data, :]
    test_x  = data_x[num_train_data:, :]
    test_y  = data_y[num_train_data:, :]

    print("Train data shape: " + str(train_x.shape))
    print("Test data shape: " + str(test_x.shape))

    plt.plot(train_x, train_y, 'b*', markersize=1)
    plt.plot(test_x, test_y, 'y*', markersize=1)
    plt.plot(data_x1, data_y1_true, 'r')
    plt.plot(data_x2, data_y2_true, 'r')
    plt.legend(['Data', 'y=x^3'], loc = 'best')
    plt.title('y = sin(x) + $\epsilon$')
    plt.show()
    return train_x, train_y, test_x, test_y

def create_toy_dataloader(batch_size, func : Callable, test_ratio=0.2):
    """
    A simple dataloader for the toy dataset.
    Args:
        batch_size (int): batch size
        test_ratio (float): ratio of test data
    
    Returns:
        train_loader (torch.utils.data.DataLoader): dataloader for training
        test_loader (torch.utils.data.DataLoader): dataloader for testing
    """
    #train_x, train_y, test_x, test_y = make_toy_dataset(test_ratio=test_ratio)
    train_x, train_y, test_x, test_y = make_toy_dataset2(test_ratio=test_ratio, func=func)
    train_dataset = ToyDataset(train_x, train_y)
    test_dataset = ToyDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_dataset, test_dataset