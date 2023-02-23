from .base import Transform
from .normalize import MinMaxNormalize, Standardize
from .to_tensor import ToTensor
from .compose import Compose

# transform registry
TRANSFORM_REGISTRY = {
    'minmax': MinMaxNormalize,
    'standardize': Standardize,
    'totensor': ToTensor,
}

def create_transform(cfg):
    """Build transforms from config"""
    transform_class = cfg.pop("class")
    if transform_class not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform : {transform_class}")
    transform = TRANSFORM_REGISTRY[transform_class]
    return transform(**cfg)
