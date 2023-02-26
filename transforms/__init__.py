from .base import Transform
from .normalize import MinMaxNormalize, Standardize
from .to_tensor import ToTensor
from .compose import Compose

# transform registry
TRANSFORM_REGISTRY = {
    'minmax': MinMaxNormalize,
    'standardize': Standardize,
    'totensor': ToTensor,
    None: Transform
}

def create_transform(cfg):
    """Build transforms from config"""
    transform_class = cfg.get("class", None)
    if transform_class not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform : {transform_class}")
    transform = TRANSFORM_REGISTRY[transform_class]
    return transform(**cfg)
