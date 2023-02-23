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
from datasets.toy_data import *

from datasets import create_dataloader
from utils import create_logger
from estimations import create_estimator
from transforms import create_transform
from transforms.compose import Compose

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
    estimator_config = config.get("estimator")
    dataset_config = config.get("dataset")
    train_config = config.get("train")
    dataset_config = config.get("dataset")
    transform_config = config.get("transforms", None)
    logger_config = config.get("logger",None)

    ######## Create dataloader ########
    logger.info("Creating dataloaders")
    train_loader, val_loader, train_ds, val_ds = create_dataloader(dataset_config)

    ######## Create estimator #########
    estimator_config["model"].update({"input_size": train_ds.x.shape[1], "output_size": train_ds.y.shape[1]})
    estimator = create_estimator(estimator_config)
    logger.info(f"Created estimator : \n\t{estimator}")

    ####### Create transforms #########
    x_transforms = Compose([create_transform(transform) for transform in transform_config.get("x", [])])
    y_transforms = Compose([create_transform(transform) for transform in transform_config.get("y", [])])

    # run the whole dataset once to get stats if mean and std is not provided
    for transform in x_transforms.transforms:
        if transform.is_pass_required:
            transform(np.concatenate([train_ds.x, val_ds.x], axis=0))
    
    for transform in y_transforms.transforms:
        if transform.is_pass_required:
            transform(np.concatenate([train_ds.y, val_ds.y], axis=0))
 
    logger.info(f"Created transforms : \n\tx: {x_transforms}\n\ty: {y_transforms}")

    ######### Create logger #########
    metric_logger = create_logger(logger_config)

    if metric_logger is not None:
        metric_logger.log_meta(config)
    logger.info(f"Created logger : \n\t{metric_logger}")

    ######### Train estimator #########
    networks = estimator.train_estimator(
        train_config = train_config,
        train_dl = train_loader,
        val_dl = val_loader,
        device = device,
        transforms = (x_transforms, y_transforms),
        logger = metric_logger)

    logger.info("Training finished")

    ######### TODO : Train predictor using uncertainity values #########


    ######### Save networks ##########
    logger.info(f"Saving networks to the path : {args.checkpoint}")
    if args.checkpoint is not None:
        logger.info(f"Saving networks to {args.checkpoint}")

        checkpoint_folder = os.path.join(args.checkpoint, time.strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        
        for i, network in enumerate(networks):
            torch.save(network.state_dict(), os.path.join(checkpoint_folder,f"network_{i}.pth"))
