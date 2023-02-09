from .toydata import ToyDataset
from .xlsdata import XLSDataset

CLASS_REGISTRY = {
    "toy": ToyDataset,
    "xls": XLSDataset,
}

def create_dataset(dataset_config):
    """
    Create a dataset based on the config file
    Args:
        dataset_config (dict): dataset config file
    Returns:
        dataset (torch.utils.data.Dataset): dataset
    """
    dataset_type = dataset_config["type"]
    if dataset_type not in CLASS_REGISTRY:
        raise ValueError("Unknown dataset type: {}".format(dataset_type))
    dataset = CLASS_REGISTRY[dataset_type](dataset_config)

    return dataset
