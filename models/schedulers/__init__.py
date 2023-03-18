import torch.optim
from copy import deepcopy

SCHEDULER_REGISTRY = {}

for name in dir(torch.optim.lr_scheduler):
    obj = getattr(torch.optim.lr_scheduler, name)
    if isinstance(obj, type) and issubclass(obj, torch.optim.lr_scheduler._LRScheduler):
        SCHEDULER_REGISTRY[name] = obj



def create_scheduler(optimizer, cfg_):
    """Build schedulers from config"""
    cfg = deepcopy(cfg_)
    scheduler_class = cfg.pop("class")
    if scheduler_class not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler : {scheduler_class}")
    return SCHEDULER_REGISTRY[scheduler_class](optimizer, **cfg)
