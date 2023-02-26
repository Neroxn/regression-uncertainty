from .logger import TensorboardLogger, WandbLogger, ClearmlLogger,PythonLogger

LOGGER_REGISTRY = {
    "tensorboard": TensorboardLogger,
    "wandb": WandbLogger,
    "clearml": ClearmlLogger,
    "default": PythonLogger
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
    