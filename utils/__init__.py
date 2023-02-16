from .logger import TensorboardLogger, WandbLogger, ClearmlLogger

LOGGER_REGISTRY = {
    "tensorboard": TensorboardLogger,
    "wandb": WandbLogger,
    "clearml": ClearmlLogger,
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
        return None
    logger_type = logger_config.pop("type")
    if logger_type not in LOGGER_REGISTRY:
        raise ValueError("Unknown logger type: {}".format(logger_type))
    logger = LOGGER_REGISTRY[logger_type](**logger_config)

    return logger
    