import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from typing import Tuple


class ToyFunc(object):
    def __init__(self, func, repr_str):
        self.func = func
        self.repr = repr_str

    def __call__(self, x):
        return self.func(x)

    def __repr__(self) -> str:
        return self.repr_str

def toy_function_sin():
    return ToyFunc(np.sin, repr_str = "sin(x)")

def toy_function_cos():
    return ToyFunc(np.cos, repr_str = "cos(x)")

def toy_function_mixed():
    def mixed(x):
        return np.sin(x) - np.cos(x/2) + np.sin(x/4) - np.sin(x/8)
    return ToyFunc(mixed, repr_str = "sin(x) + cos(x/2) + sin(x/4) + 3*sin(x/8)")

def toy_function_complex():
    def complex(x):
        return  np.exp(np.cos(3*x**2 + 4*x)) + np.sin(x**3) + np.sin(x) - 2*np.cos(x/2) + 4*np.sin(x/4) - 8*np.sin(np.sin(x/8) + np.sin(x/16)) + np.exp(np.sin(x/32))
    return ToyFunc(complex, repr_str = "x^2 - 3x + sin(x^3) + sin(x) - 2cos(x/2) + 4sin(x/4) - 8sin(sin(x/8) + sin(x/16)) + exp(sin(x/32))")
TOY_FUNC_REGISTRY = {
    "toy_function_sin": toy_function_sin,
    "toy_function_cos": toy_function_cos,
    "toy_function_mixed": toy_function_mixed,
    "toy_function_complex": toy_function_complex
}

class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def normalize(self):
        """
        Normalize the data
        """
        self.mean_x = np.mean(self.x, axis=0)
        self.std_x = np.std(self.x, axis=0)
        self.mean_y = np.mean(self.y, axis=0)
        self.std_y = np.std(self.y, axis=0)

        self.x = (self.x - np.mean(self.x, axis=0)) / np.std(self.x, axis=0)
        self.y = (self.y - np.mean(self.y, axis=0)) / np.std(self.y, axis=0)

    def denormalize(self, x, y):
        """
        Denormalize the data
        """
        x = x * self.std_x + self.mean_x
        y = y * self.std_y + self.mean_y
        return x, y

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

def make_toy_dataset(
    func : Callable, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    data_range = kwargs.get("data_range", 7)
    data_step = kwargs.get("data_step", 0.001)
    bounds = kwargs.get("bounds", (-2, 2))
    sigmas = kwargs.get("sigmas", (0.1, 0.5))
    test_ratio = kwargs.get("test_ratio", 0.2)

    bound1, bound2 = bounds
    data_sigma1, data_sigma2 = sigmas

    data_x1 = np.arange(-data_range, bound1 + data_step, data_step)
    data_x2 = np.arange(bound2, data_range + data_step, data_step)

    data_x = np.concatenate((data_x1, data_x2))
    data_x = np.reshape(data_x, [data_x.shape[0], 1])

    data_y = np.zeros([data_x.shape[0], 1])

    for i in range(data_x.shape[0]):
        if (data_x[i,0] < bound1): 
            data_y[i, 0] = func(data_x[i,0]) + np.random.normal(0, data_sigma1)
        else:
            data_y[i, 0] = func(data_x[i,0]) + np.random.normal(0, data_sigma2)

    data_x, data_y = shuffle(data_x, data_y)
            
    num_train_data = int(data_x.shape[0] * (1 - test_ratio))

    train_x = data_x[:num_train_data, :]
    train_y = data_y[:num_train_data, :]
    test_x  = data_x[num_train_data:, :]
    test_y  = data_y[num_train_data:, :]

    return train_x, train_y, test_x, test_y

def create_toy_dataloader(**kwargs):
    """
    A simple dataloader for the toy dataset.
    Args:
        batch_size (int): batch size
        test_ratio (float): ratio of test data
    
    Returns:
        train_loader (torch.utils.data.DataLoader): dataloader for training
        test_loader (torch.utils.data.DataLoader): dataloader for testing
    """
    func = kwargs.pop("func", None)
    if func is None:
        raise ValueError("Please provide a function to sample from.")
    func = TOY_FUNC_REGISTRY[func]()

    batch_size = kwargs.pop("batch_size", 32)
    train_x, train_y, test_x, test_y = make_toy_dataset(func = func, **kwargs)
    train_dataset = ToyDataset(train_x, train_y)
    test_dataset = ToyDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_dataset, test_dataset, None, None


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
