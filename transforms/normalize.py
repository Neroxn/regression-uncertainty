import numpy as np 
from typing import Tuple, Union
import torch
import transforms.base 

ArrayLike = Union[np.ndarray, torch.Tensor, list]

class MinMaxNormalize(transforms.base.Transform):
    def __init__(self, min_val : ArrayLike, max_val : ArrayLike):
        super(MinMaxNormalize, self).__init__()
        if isinstance(min_val, list):
            min_val = np.array(min_val)
        if isinstance(max_val, list):
            max_val = np.array(max_val)
        if isinstance(min_val, np.ndarray):
            min_val = torch.from_numpy(min_val)
        if isinstance(max_val, np.ndarray):
            max_val = torch.from_numpy(max_val)

        self.min_val = min_val
        self.max_val = max_val


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            
        # check if all tensors on the same device
        if x.device != self.min_val.device:
            self.min_val = self.min_val.to(x.device)
            self.max_val = self.max_val.to(x.device)

        x = (x - self.min_val) / (self.max_val - self.min_val)

        return x


    def backward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # check if all tensors on the same device
        if x.device != self.min_val.device:
            self.min_val = self.min_val.to(x.device)
            self.max_val = self.max_val.to(x.device)

        x = x * (self.max_val - self.min_val) + self.min_val
        return x

    def __str__(self) -> str:
        return f"MinMaxNormalize(min_val={self.min_val}, max_val={self.max_val})"

class Standardize(transforms.base.Transform):
    def __init__(self, mean : ArrayLike, std : ArrayLike):
        super(Standardize, self).__init__()

        if isinstance(mean, list):
            mean = np.array(mean)
        if isinstance(std, list):
            std = np.array(std)
        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std)

        self.mean = mean
        self.std = std


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # check if all tensors on the same device
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        x = (x - self.mean) / self.std

        return x

    def backward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # check if all tensors on the same device
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        x = x * self.std + self.mean
        return x

    def __str__(self) -> str:
        return f"Standardize(mean={self.mean}, std={self.std})"

