import torch 
from .base import BaseModel
from .multi_layer import LinearVarianceNetworkHead, Conv2DVarianceNetworkHead
from copy import deepcopy

# Custom layer registery. Custom layers can be added to this registery.
LAYER_REGISTERY = {
    "LinearVarianceNetworkHead": LinearVarianceNetworkHead,
    "Conv2DVarianceNetworkHead": Conv2DVarianceNetworkHead,
} 

# Add all torch.nn layers to the registery
for name in dir(torch.nn):
    obj = getattr(torch.nn, name)
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
        LAYER_REGISTERY[name] = obj

def create_model(model_config):
    """
    Build a model from a config file. The config file is a list of dictionaries.
    Each dictionary contains the name of the layer and the arguments for the layer.
    """
    base_model = BaseModel()
    model_config = deepcopy(model_config)
    for layer_config in model_config:
        for layer_name, layer_args in layer_config.items():
            layer_class = layer_args.pop("class")
            layer = LAYER_REGISTERY[layer_class](**layer_args)
            base_model.add_layer(layer_name, layer)
    return base_model