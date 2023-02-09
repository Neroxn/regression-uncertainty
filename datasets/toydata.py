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

    # plt.plot(train_x, train_y, 'b*', markersize=1)
    # plt.plot(test_x, test_y, 'y*', markersize=1)
    # plt.plot(data_x1, data_y1_true, 'r')
    # plt.plot(data_x2, data_y2_true, 'r')
    # plt.legend(['Data', 'y=x^3'], loc = 'best')
    # plt.title('y = sin(x) + $\epsilon$')
    # plt.show()
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


def sample_toy_data(func : Callable, start : float, end : float, step : float):
    """
    Using a function, sample data between two points.

    Arguments:
        - func (Callable): the function to sample from
        - start (float): the start of the range
        - end (float): the end of the range
        - step (float): the step size
    
    Returns:
        - x (np.ndarray): the x values
        - y (np.ndarray): the y values
    """
    x_sample = np.arange(start, end + step, step)
    x_sample = np.reshape(x_sample, [x_sample.shape[0], 1]) # reshape to column vector
    y_sample = func(x_sample)
    return x_sample, y_sample


def plot_toy_results(
    x_sample : np.ndarray,
    y_sample : np.ndarray,
    train_ds : Dataset,
    data_mu : np.ndarray,
    data_sig : np.ndarray,
    save_path : str = None):
    """
    Plot the mu/variance results for the toy dataset.
    Arguments:
        - x_sample (np.ndarray): the x values
        - y_sample (np.ndarray): the y values
        - train_ds (Dataset): the training dataset
        - data_mu (np.ndarray): the mean predictions
        - data_sig (np.ndarray): the variance predictions
        - save_path (str) : the path to save the plot to
    """
    plt.figure(1)
    plt.plot(train_ds.x, train_ds.y, 'b*', label='train data', zorder=1, alpha=0.5)
    plt.plot(x_sample, y_sample, 'r', linewidth = 2, label='y=sin(x)', zorder=1)
    plt.plot(x_sample, data_mu, 'g', linewidth = 2, label='mean', zorder=1)
    plt.fill_between(x_sample, data_mu - data_sig, data_mu + data_sig, color='g', alpha=0.4, label = "variance", zorder=2)
    plt.legend()
    plt.savefig(save_path)
