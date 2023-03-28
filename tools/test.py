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
        '--checkpoint',
        help='Checkpoint to save the model',
        default=None)
    parser.add_argument(
        '--load_estimator_from',
        help='Load the model from a checkpoint', required=True)
    parser.add_argument(
        '--load_predictor_from',
        help='Load the model from a checkpoint', required=True)
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
    for i,(_, val_loader, test_loader) in enumerate(dl_split_iterator):

        ######## Create estimator #########
        estimator = create_estimator(estimator_config, num_networks=estimator_config.get("num_networks",0))
        logger.info(f"Created estimator : \n\t{estimator.estimators}")

        ######### Load uncertainity estimator #########
        if args.load_estimator_from:
            estimator.init_estimator("estimator_network")
            networks = estimator.estimators

            for i, network in enumerate(networks):
                network.load_state_dict(torch.load(os.path.join(args.load_from, f"estimator_network_{i}.pth")))
                network.to(device)
            
        ######### Load predictor #########
        if args.load_predictor_from:
            estimator.init_predictor("predictor_network")
            predictor = estimator.predictor
            logger.log(f"Loading predictor from {args.load_from}")
            predictor.load_state_dict(torch.load(os.path.join(args.load_from, f"predictor_network.pth")))
            predictor.to(device)
            
        ######### Test the predictor #########
        val_mu, val_true = estimator.evaluate_predictor(val_loader, predictor, transforms, device)
        val_metric_list = metrics.get("val", None)
        if val_metric_list is not None:
            if categorize:
                assert hasattr(val_loader.dataset, "get_categories"), "Dataset class must have get_categories method"
                categories = val_loader.dataset.get_categories()

                # get category labels for all val_true
                val_true_cat = np.array([categories.get(int(val_true[i]),"zero-shot") for i in range(len(val_true))])
                for cat in np.unique(val_true_cat):
                    cat_mask = val_true_cat == cat
                    cat_val_true = val_true[cat_mask]
                    cat_val_mu = val_mu[cat_mask]

                    val_metric_list.forward(cat_val_mu, cat_val_true)
                    logger.log_metric(
                        metric_dict = val_metric_list.get_metrics(add_prefix = "predictor_val_" + cat),
                        step = 0,
                    )
            val_metric_list.forward(val_mu, val_true)
            logger.log_metric(
                metric_dict = val_metric_list.get_metrics(add_prefix = "predictor_val"),
                step = 0)

    for key, value in cv_track_list.items():
        metric_mean = np.mean(value)
        metric_std = np.std(value)
        logger.info( f"{key} : {metric_mean:.3f} ± {metric_std:.3f}")

    logger.info("Training finished")


