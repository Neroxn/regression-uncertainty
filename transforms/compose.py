import torch
import numpy as np
import transforms.base
from typing import Tuple, List

class Compose(transforms.base.Transform):
    def __init__(self, transforms : List[transforms.base.Transform]):
        self.transforms = transforms

    def forward(self, x):
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def backward(self, x):
        for transform in self.transforms[::-1]:
            x = transform.backward(x)
        return x

