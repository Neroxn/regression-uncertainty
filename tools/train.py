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

from datasets.toydata import create_toy_dataloader, sample_toy_data, plot_toy_results
from datasets.toyfunc import toy_func_mixed
from datasets.xlsdata import create_xls_dataloader

from utils import create_logger

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
    parser.add_argument(
        '--seed',
        help='Random seed to use. Use a random seed if not provided',
        default=42)
    parser.add_argument(
        '--save_dir',
        help='Directory to save the results',
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
    logger_config = config["logger"]



    # create dataloaders
    logger.info("Creating dataloaders")
    if dataset_config.get("class", None) == "toy":
        toy_func = toy_func_mixed()
        train_loader, val_loader, train_ds, val_ds = create_toy_dataloader(
            train_config["batch_size"],
            toy_func,
        )
        x_sample, y_sample = sample_toy_data(toy_func, -7, 7, 0.1)

    elif dataset_config.get("class", None) == "xls":
        train_loader, val_loader, train_ds, val_ds, x_stats, y_stats = create_xls_dataloader(
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

    # setup metric logger
    metric_logger = create_logger(logger_config)
    logger.info(f"Created metric logger : {metric_logger}")

    # train the networks
    logger.info("Training networks")
    networks = train_multiple_networks(
        train_loader,
        val_loader,
        networks,
        optimizers,
        device,
        num_iter = train_config["num_iter"],
        print_every = train_config["print_every"],
        logger = metric_logger,
    )

    logger.info("Testing networks")
    data_mu, data_sig = test_multiple_networks(
        x_sample,
        networks,
        device,
        logger = metric_logger
    )

if dataset_config.get("class", None) == "toy":
    plot_toy_results(
        x_sample.reshape(-1),
        y_sample.reshape(-1),
        train_ds,
        data_mu,
        data_sig,
        save_path=args.save_dir
    )
else:
    pass