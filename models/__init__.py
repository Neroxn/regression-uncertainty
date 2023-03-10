import torch 
from .base import BaseModel
from .multi_layer import LinearVarianceNetworkHead, Conv2DVarianceNetworkHead
from .resnet import BasicBlock, Bottleneck, ResNet
from copy import deepcopy

# Custom layer registery. Custom layers can be added to this registery.
LAYER_REGISTERY = {
    "LinearVarianceNetworkHead": LinearVarianceNetworkHead,
    "Conv2DVarianceNetworkHead": Conv2DVarianceNetworkHead,
    "BasicBlock" : BasicBlock,
    "Bottleneck" : Bottleneck,
    "ResNet" : ResNet,
} 

# Add all torch.nn layers to the registery
for name in dir(torch.nn):
    obj = getattr(torch.nn, name)
    if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
        LAYER_REGISTERY[name] = obj

def _replace_layer(value):
    """
    Replace a value that is an instance of list or str with the corresponding layer.
    """
    if value.__class__ == list:
        return [_replace_layer(v) for v in value]
    elif value.__class__ == str:
        if value in LAYER_REGISTERY:
            return LAYER_REGISTERY[value]
    return value

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

            # layer_args might include another dict. Check recursively for possible block constructions
            for key, value in layer_args.items():
                if value.__class__ == dict and  value.get("class"): 
                    layer_args[key] = create_model([value]) # construct the layer
                layer_args[key] = _replace_layer(value) # replace the layer with the actual layer class
            layer = LAYER_REGISTERY[layer_class](**layer_args)
            base_model.add_layer(layer_name, layer)
    return base_model
