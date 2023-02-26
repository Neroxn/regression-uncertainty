from .toy_data import *
from .xls_data import *

# Add to the following class as the list goes on 
CLASS_REGISTRY = {
    "toy": ToyDataset,
    "xls": XLSDataset,
}

DATA_LOADER_REGISTRY = {
    "toy" : create_toy_dataloader,
    "xls" : create_xls_dataloader,
}

def create_dataset(dataset_config):
    """
    Create a dataset based on the config file
    Args:
        dataset_config (dict): dataset config file
    Returns:
        dataset (torch.utils.data.Dataset): dataset
    """
    dataset_type = dataset_config["class"]
    if dataset_type not in CLASS_REGISTRY:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))
    dataset = CLASS_REGISTRY[dataset_type](**dataset_config)

    return dataset

def create_dataloader(dataset_config):
    """
    Create a dataloader based on the config file
    """
    dataset_type = dataset_config["class"]
    if dataset_type not in DATA_LOADER_REGISTRY:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))
    return DATA_LOADER_REGISTRY[dataset_type](**dataset_config)