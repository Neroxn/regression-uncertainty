import argparse
import yaml

import sys 

import matplotlib.pyplot as plt 
# append path so that modules can be imported
sys.path.append('.')

import numpy as np

from utils.logger import set_logger, pretty_dict_print
from utils.device import set_device, set_random

from estimations.ensemble import (VarianceNetwork, create_multiple_networks, train_multiple_networks
                                  ,test_multiple_networks)

from datasets.toydata import create_toy_dataloader
from datasets.toyfunc import toy_func_mixed
from datasets.concrete import create_xls_dataloader

def parse_config(config_path):
    """
    Parse a .yaml file to a dictionary

    Args: 
        config_path (str): path to the config file
    
    Returns:
        dict: a dictionary of the config file
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
def parse_args():
    """
    Parse arguments from command line
    """
    parser = argparse.ArgumentParser(description='Train networks by using uncertainty information')
    parser.add_argument('--config', help='train config file path', required=True)
    parser.add_argument(
        '--device',
        help='Device to use. Use available device in the order of cuda, metal and cpu if not provided',
        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger = set_logger()
    device = set_device(args.device)
    logger.info(f"Using device : {device}")
    set_random()

    logger.info("Parsing config file")
    config = parse_config(args.config)

    pretty_dict_print(config)
    network_config = config["network"]
    dataset_config = config["dataset"]
    train_config = config["train"]
    dataset_config = config["dataset"]

    # create dataloaders
    logger.info("Creating dataloaders")
    if dataset_config.get("class", None) == "ToyData":
        toy_func = toy_func_mixed()
        train_loader, val_loader, train_ds, val_ds = create_toy_dataloader(
            train_config["batch_size"],
            toy_func,
        )
    elif dataset_config.get("class", None) == "XLSData":
        train_loader, val_loader, train_ds, val_ds = create_xls_dataloader(
            train_config["batch_size"],
            dataset_config["xls_path"]
        )
    else:
        raise ValueError(f"Unknown dataset class {dataset_config.get('class', None)}")
        
    # create multiple networks
    input_size = train_ds.x.shape[1]
    output_size = train_ds.y.shape[1]


    num_networks = network_config["num_networks"]
    networks, optimizers = create_multiple_networks(
        num_networks,
        input_size,
        network_config["layer_sizes"],
        output_size
    )
    logger.info(f"Created network : {networks[-1]}")

    # train the networks
    logger.info("Training networks")
    networks = train_multiple_networks(
        train_loader,
        val_loader,
        networks,
        optimizers,
        device,
        num_iter = train_config["num_iter"],
        print_every = train_config["print_every"]
    )

    # test the networks
    x_sample = np.arange(-12, 12 + 0.1, 0.1)

    x_sample = np.reshape(x_sample, [x_sample.shape[0], 1])

    logger.info("Testing networks")
    data_mu, data_sig = test_multiple_networks(
        x_sample,
        networks,
        device
    )

    plt.figure(1)

    # shape x_sample to be [num_samples,] 
    x_sample = x_sample.reshape(-1)
    y_sample = toy_func(x_sample)

    plt.plot(train_ds.x, train_ds.y, 'b*', label='train data', zorder=1, alpha=0.5)
    plt.plot(x_sample, y_sample, 'r', linewidth = 2, label='y=sin(x)', zorder=1)
    plt.plot(x_sample, data_mu, 'g', linewidth = 2, label='mean', zorder=1)
    plt.fill_between(x_sample, data_mu - data_sig, data_mu + data_sig, color='g', alpha=0.4, label = "variance", zorder=2)

    plt.legend()
    plt.show()