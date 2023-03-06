import torch.optim

from .qhadam import QHAdam
from copy import deepcopy

OPTIMIZER_REGISTRY = {"QHAdam" : QHAdam}

# Add all torch.optim optimizers to the registry
for name in dir(torch.optim):
    obj = getattr(torch.optim, name)
    if isinstance(obj, type) and issubclass(obj, torch.optim.Optimizer):
        OPTIMIZER_REGISTRY[name] = obj

def create_optimizer(params, cfg_):
    """Build optimizer from config"""
    cfg = deepcopy(cfg_)
    optimizer_class = cfg.pop("class")
    if optimizer_class not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer : {optimizer_class}")
    return OPTIMIZER_REGISTRY[optimizer_class](params = params, **cfg)
