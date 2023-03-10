import torchvision
from .base import Transform,PytorchWrapper
from .normalize import MinMaxNormalize, Standardize
from .compose import Compose
import torch 

from copy import deepcopy

# transform registry
TRANSFORM_REGISTRY = {
    'MinMaxNormalize': MinMaxNormalize,
    'Standardize': Standardize,
}

PYTORCH_REGISTRY  = {}

# add pytorch transforms to the registry. 
for name in dir(torchvision.transforms):
    obj = getattr(torchvision.transforms, name)
    if isinstance(obj, type):
        PYTORCH_REGISTRY[name] = obj

def create_transform(cfg_):
    """Build transforms from config"""
    cfg = deepcopy(cfg_)
    transform_class = cfg.pop("class", None)
    if transform_class not in TRANSFORM_REGISTRY:
        if transform_class in PYTORCH_REGISTRY:
            return PytorchWrapper(PYTORCH_REGISTRY[transform_class](**cfg))
        raise ValueError(f"Unknown transform : {transform_class}")
    return TRANSFORM_REGISTRY[transform_class](**cfg)
