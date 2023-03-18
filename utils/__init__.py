from .logger import TensorboardLogger, WandbLogger, ClearmlLogger,PythonLogger
from .metrics import *

LOGGER_REGISTRY = {
    "tensorboard": TensorboardLogger,
    "wandb": WandbLogger,
    "clearml": ClearmlLogger,
    "default": PythonLogger
}

METRIC_REGISTRY = {
    "rmse" : rmse,
    "mae" : mae,
    "mape" : mape,
    "mse" : mse,
    "gm" : gm,
    "nll" : nll,
    "nll_mse" : nll_mse,
    "weighted_mse" : weighted_mse,
    "weighted_rmse" : weighted_rmse,
    "weighted_mae" : weighted_mae,
    "pearson_correlation" : pearson_correlation
}


def create_logger(logger_config):
    """
    Create a logger based on the config file
    Args:
        logger_config (dict): logger config file
    Returns:
        logger (NetworkLogger): logger
    """
    if logger_config is None:
        logger_type = "default"
        logger = LOGGER_REGISTRY[logger_type]()
    else:
        logger_type = logger_config.pop("type")
        if logger_type not in LOGGER_REGISTRY:
            print("Unknown logger type: {}. Using default python logger".format(logger_type))
            logger_type = "default"
        logger = LOGGER_REGISTRY[logger_type](**logger_config)
    return logger
    

def create_metric_list(metric_list):
    """
    Create a list of metrics based on the config file
    Args:
        metric_list (list): list of metrics
    Returns:
        metric_list (list): list of metrics
    """
    metrics = []
    for metric in metric_list:
        if metric not in METRIC_REGISTRY:
            raise ValueError("Unknown metric type: {}".format(metric))
        metric_fn = METRIC_REGISTRY[metric]
        metrics.append(Metric(metric_fn, metric))
    return MetricList(metrics)