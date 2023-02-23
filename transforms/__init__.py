from .base import Transform
from .normalize import MinMaxNormalize, Standardize
from .to_tensor import ToTensor
from .compose import Compose

# transform registry
transforms = {
    'minmax': MinMaxNormalize,
    'standardize': Standardize,
    'totensor': ToTensor,
}

def build_transforms(cfg):
    """Build transforms from config"""
    transforms_list = []
    for transform_name in cfg:
        transform = transforms[transform_name]
        transforms_list.append(transform(**cfg[transform_name]))
    return Compose(transforms_list)
    