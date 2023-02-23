
import numpy as np 
from typing import Tuple
import torch
import transforms.base 

class ToTensor(transforms.base.Transform):
    def __init__(self, dtype = torch.float32):
        self.dtype = dtype

    def forward(self, x : np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x).type(self.dtype)
        return x
        
    def backward(self, x : torch.Tensor) -> np.ndarray:
        x = x.numpy()
        return x

        

    
    
