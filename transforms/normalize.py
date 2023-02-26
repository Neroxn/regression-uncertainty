import numpy as np 
from typing import Tuple
import torch
import transforms.base 


class MinMaxNormalize(transforms.base.Transform):
    def __init__(self, x_min : torch.Tensor = None, x_max : torch.Tensor = None, **kwargs):
        self.is_pass_required = True if x_min is None and x_max is None else False

        if isinstance(x_min, np.ndarray):
            x_min = torch.from_numpy(x_min)
        if isinstance(x_max, np.ndarray):
            x_max = torch.from_numpy(x_max)

        self.x_min = x_min
        self.x_max = x_max


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if self.x_min is None:
            self.x_min = torch.min(x, axis=0)

        if self.x_max is None:
            self.x_max = torch.max(x, axis=0)
            
        # check if all tensors on the same device
        if x.device != self.x_min.device:
            self.x_min = self.x_min.to(x.device)
            self.x_max = self.x_max.to(x.device)

        x = (x - self.x_min) / (self.x_max - self.x_min)

        return x


    def backward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x * (self.x_max - self.x_min) + self.x_min

        # check if all tensors on the same device
        if x.device != self.x_min.device:
            self.x_min = self.x_min.to(x.device)
            self.x_max = self.x_max.to(x.device)
        return x

    def __str__(self) -> str:
        return f"MinMaxNormalize(x_min={self.x_min}, x_max={self.x_max})"

class Standardize(transforms.base.Transform):
    def __init__(self, x_mean : torch.Tensor = None, x_std : torch.Tensor = None, **kwargs):
        self.is_pass_required = True if x_mean is None and x_std is None else False

        if isinstance(x_mean, np.ndarray):
            x_mean = torch.from_numpy(x_mean)
        if isinstance(x_std, np.ndarray):
            x_std = torch.from_numpy(x_std)

        self.x_mean = x_mean
        self.x_std = x_std


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if self.x_mean is None:
            self.x_mean = torch.mean(x, axis=0)

        if self.x_std is None:
            self.x_std = torch.std(x, axis=0)
        
        # check if all tensors on the same device
        if x.device != self.x_mean.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)

        x = (x - self.x_mean) / self.x_std

        return x

    def backward(self, x : torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # check if all tensors on the same device
        if x.device != self.x_mean.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)

        x = x * self.x_std + self.x_mean
        return x

    def __str__(self) -> str:
        return f"Standardize(x_mean={self.x_mean}, x_std={self.x_std})"

