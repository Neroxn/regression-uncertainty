import numpy as np 
from typing import Tuple
import torch
import transforms.base 


class MinMaxNormalize(transforms.base.Transform):
    def __init__(self, x_min : float = None, x_max : float = None):
        self.x_min = x_min
        self.x_max = x_max


    def forward(self, x : np.ndarray) -> np.ndarray:
        if self.x_min is None:
            self.x_min = np.min(x)

        if self.x_max is None:
            self.x_max = np.max(x)
            
        x = (x - self.x_min) / (self.x_max - self.x_min)

        return x


    def backward(self, x : np.ndarray) -> np.ndarray:
        x = x * (self.x_max - self.x_min) + self.x_min
        return x

class Standardize(transforms.base.Transform):
    def __init__(self, x_mean : float = None, x_std : float = None):
        self.x_mean = x_mean
        self.x_std = x_std


    def forward(self, x : np.ndarray) -> np.ndarray:
        if self.x_mean is None:
            self.x_mean = np.mean(x)

        if self.x_std is None:
            self.x_std = np.std(x)
        
        x = (x - self.x_mean) / self.x_std

        return x

    def backward(self, x : np.ndarray) -> np.ndarray:
        x = x * self.x_std + self.x_mean
        return x


