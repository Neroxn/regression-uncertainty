import torchvision
from .base import Transform,PytorchWrapper
from .normalize import MinMaxNormalize, Standardize
from .compose import Compose
import warnings
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

def get_transforms(cfg):
    """Build transforms from config"""
    transforms = {}
    transforms_train = cfg.get("train", [])
    transforms_val = cfg.get("val", [])
    transforms_test = cfg.get("test", [])

    if transforms_test == []:
        warnings.warn("Test transforms not specified. Using val transforms for test.")
        transforms_test = transforms_val

    transforms["train"] = {
        "x" : Compose([create_transform(transform) for transform in transforms_train.get("x", [])]),
        "y" : Compose([create_transform(transform) for transform in transforms_train.get("y", [])])}
    transforms["val"] = {
        "x" : Compose([create_transform(transform) for transform in transforms_val.get("x", [])]),
        "y" : Compose([create_transform(transform) for transform in transforms_val.get("y", [])])}
    transforms["test"] = {
        "x" : Compose([create_transform(transform) for transform in transforms_test.get("x", [])]),
        "y" : Compose([create_transform(transform) for transform in transforms_test.get("y", [])])}
    
    return transforms