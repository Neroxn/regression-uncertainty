import argparse
import yaml

import sys 
import os 

import matplotlib.pyplot as plt 
# append path so that modules can be imported
sys.path.append('.')

import numpy as np
import torch 
import time

from utils.logger import set_logger, pretty_dict_print
from utils.device import set_device, set_random

from estimations.ensemble import create_multiple_networks, train_multiple_networks, test_multiple_networks

from datasets.toy_data import *
from datasets import create_dataloader
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
    parser.add_argument(
        '--checkpoint',
        help='Checkpoint to load the model',
        default=None)

    return parser.parse_args()


if __name__ == '__main__':

    # get args 
    args = parse_args()

    # set random seed for reproducibility
    set_random()
    logger = set_logger()
    device = set_device(args.device)
    logger.info(f"Using device : {device}")

    # parse config file
    config = parse_config(args.config)
    network_config = config.get("network")
    dataset_config = config.get("dataset")
    train_config = config.get("train")
    dataset_config = config.get("dataset")
    logger_config = config.get("logger",None)
    pretty_dict_print(config)

    # create dataloaders
    logger.info("Creating dataloaders")
    train_loader, val_loader, train_ds, val_ds, x_stats, y_stats = create_dataloader(dataset_config)

    # create ensamble of networks
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

    # log the config file
    if metric_logger is not None:
        metric_logger.log_meta(config)

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
        weighted = train_config["weighted"],
        logger = metric_logger,
        ds_stats = (x_stats, y_stats) if y_stats is not None else None
    )
    
if dataset_config.get("class", None) == "toy":
    func = dataset_config.get("func", None)
    func = TOY_FUNC_REGISTRY[func]()
    x_sample, y_sample = sample_toy_data(func, -10, 10, 0.1)
    data_mu, data_sig = test_multiple_networks(
        x_sample,
        networks,
        device,
    )

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
    # save the networks
    logger.info(f"Saving networks to the path : {args.checkpoint}")
    if args.checkpoint is not None:
        logger.info(f"Saving networks to {args.checkpoint}")
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
        for i, network in enumerate(networks):
            torch.save(network.state_dict(), os.path.join(args.checkpoint,f"network_{i}.pth"))
