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
from utils import create_logger, create_metric_list
from estimations import create_estimator
from transforms import get_transforms
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
        default=42,
        type = int
        )
    parser.add_argument(
        '--save_dir',
        help='Directory to save the results',
        default=None)
    parser.add_argument(
        '--checkpoint',
        help='Checkpoint to load the model',
        default=None)
    parser.add_argument(
        '--load_from',
        help='Load the model from a checkpoint',
        default=None)
    parser.add_argument(
        '--weighted_training',
        help='Use weighted training',
        action='store_true',
        default=False)
    parser.add_argument(
        '--only_estimators',
        action='store_true'
    )
    return parser.parse_args()


if __name__ == '__main__':
    ######### Parse config and set random #########
    args = parse_args()
    device = set_device(args.device)
    config = parse_config(args.config)
    estimator_config = config.get("estimator")
    dataset_config = config.get("dataset")
    train_config = config.get("train")
    dataset_config = config.get("dataset")
    transform_config = config.get("transforms", None)
    logger_config = config.get("logger",None)
    metric_config = config.get("metrics",None)

    if args.seed != -1:
        set_random(args.seed)
    
    ######### Create logger #########
    metric_logger = create_logger(logger_config)
    if logger_config is None:
        logger = metric_logger.logger
    else:
        logger = set_logger()

    pretty_dict_print(config)

    if metric_logger is not None:
        metric_logger.log_meta(config)
    logger.info(f"Created logger : \n\t{metric_logger}")
    logger.info(f"Using device : {device}")
    
    ######### Create metrics #########
    metrics = {}
    if metric_config is not None:
        metric_config_list = metric_config.get("list")
        metrics["train"] = create_metric_list(metric_config_list.get("train", []))
        metrics["val"] = create_metric_list(metric_config_list.get("val", []))
        metrics["test"] = create_metric_list(metric_config_list.get("test",[]))

    categorize = metric_config.get("categorize", False)
    ######### Create Transforms #########
    transforms = get_transforms(transform_config)
    

    ######## Create Dataloader ########
    dl_split_iterator = create_dataloader(dataset_config, transforms) # yields train_loader, val_loader, train_ds, val_ds
    logger.info(f"Created dataloader : \n\t{dl_split_iterator}")

    ######### Start Cross-Validation (Optional, if cv_split_num is 1, it is a regular training) #########
    cv_track_list = {}
    for i,(train_loader, val_loader, test_loader) in enumerate(dl_split_iterator):

        ######## Create estimator #########
        estimator = create_estimator(estimator_config, num_networks=estimator_config.get("num_networks",0))
        logger.info(f"Created estimator : \n\t{estimator}")

        ######### Train uncertainty estimator #########
        if args.load_from is None:
            networks = estimator.train_estimator(
                train_config = train_config,
                train_dl = train_loader,
                val_dl = val_loader,
                device = device,
                transforms = transforms,
                logger_prefix = f"estimator",
                logger = metric_logger,
                metrics = metrics,
                categorize = categorize,
                checkpoint = args.checkpoint
                )
            
            if args.checkpoint is not None:
                logger.info(f"Saving networks to {args.checkpoint}")

                checkpoint_folder = os.path.join(args.checkpoint, time.strftime("%Y%m%d-%H%M%S"))
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                
                for i, network in enumerate(networks):
                    torch.save(network.state_dict(), os.path.join(checkpoint_folder,f"estimator_network_{i}.pth"))
        else:
            estimator.init_estimator("estimator_network")
            networks = estimator.estimators

            logger.info(f"Loading estimators from {args.load_from}")
            for i, network in enumerate(networks):
                network.load_state_dict(torch.load(os.path.join(args.load_from, f"estimator_network_{i}.pth")))
                network.to(device)

        ######## Train final predictor network #########
        if not args.only_estimators:
            if args.load_from is None:
                print("Training predictor network")
                predictor = estimator.train_predictor(
                    weighted_training=args.weighted_training,
                    train_config = train_config,
                    train_dl = train_loader,
                    val_dl = val_loader,
                    device = device,
                    transforms = transforms,
                    logger = metric_logger,
                    logger_prefix = f"predictor",
                    metrics = metrics,
                    categorize = categorize
                    )
            else:
                estimator.init_predictor("predictor_network")
                predictor = estimator.predictor
                logger.log(f"Loading predictor from {args.load_from}")
                predictor.load_state_dict(torch.load(os.path.join(args.load_from, f"predictor_network.pth")))
                predictor.to(device)
            
            ######### Save networks ##########
            if args.checkpoint is not None:
                checkpoint_folder = os.path.join(args.checkpoint, time.strftime("%Y%m%d-%H%M%S"))
                torch.save(predictor.state_dict(), os.path.join(checkpoint_folder,f"predictor_network.pth"))

        track_list = metric_logger.reset_track_list()
        for key, value in track_list.items():
            if key not in cv_track_list:
                cv_track_list[key] = []
            cv_track_list[key].append(value[-1])

    for key, value in cv_track_list.items():
        metric_mean = np.mean(value)
        metric_std = np.std(value)
        logger.info( f"{key} : {metric_mean:.3f} ± {metric_std:.3f}")

    logger.info("Training finished")


