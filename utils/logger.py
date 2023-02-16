import logging

LOGGER_CHECK = {}
try:
    import wandb
    LOGGER_CHECK["wandb"] = True
except ImportError:
    LOGGER_CHECK["wandb"] = False

try:
    import clearml
    LOGGER_CHECK["clearml"] = True
except ImportError:
    LOGGER_CHECK["clearml"] = False
    pass

try:
    from torch.utils.tensorboard import SummaryWriter
    LOGGER_CHECK["tensorboard"] = True
except ImportError:
    LOGGER_CHECK["tensorboard"] = False
    pass


def pretty_dict_print(dictionary : dict):
    """
    Print a dictionary in a pretty way
    """
    print("--------------------")
    for key, value in dictionary.items():
        print('%s: %s' % (key, value))
    print("--------------------")

def set_logger():
    """
    Set the logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create error file handler and set level to error
    fh = logging.FileHandler("error.log")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # create debug file handler and set level to debug
    fh = logging.FileHandler("debug.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

class NetworkLogger(object):
    """ Base class for logging network training. """

    def __init__(self):
        self.track_list = {}
    def log_metric(self, **kwargs):
        """ Log a metric using the logger."""
        raise NotImplementedError

    def init_logger(self):
        """ Initialize the logger. """
        raise NotImplementedError


class WandbLogger(NetworkLogger):
    """ Logger for logging network training to wandb. """
    def __init__(self, **kwargs):
        if LOGGER_CHECK["wandb"] is False:
            raise ImportError("wandb is not installed")
        self.init_logger(**kwargs)

    def log_metric(self, metric_dict, step):
        wandb.log(metric_dict, step=step)

    def init_logger(self, **kwargs):
        return wandb.init(**kwargs)

    def log_meta(self, meta_dict):
        wandb.config.update(meta_dict)

class ClearmlLogger(NetworkLogger):
    """ Logger for logging network training to clearml. """
    def __init__(self, **kwargs):
        if LOGGER_CHECK["clearml"] is False:
            raise ImportError("clearml is not installed")
        self.init_logger(**kwargs)

    def init_logger(self, **kwargs):
        self.task = clearml.Task.init(**kwargs)
        self.logger = self.task.get_logger()

    def log_metric(self, metric_dict, step):
        for k,v in metric_dict.items():
            self.logger.report_scalar(title=k, series=k, value=v, iteration=step)

    def log_matplotlib_figure(self, figure, figure_name, step):
        self.logger.report_matplotlib_figure(figure=figure, figure_name=figure_name, iteration=step)
        
    def log_meta(self, meta_dict):
        self.task.connect(meta_dict)

class TensorboardLogger(NetworkLogger):
    """ Logger for logging network training to tensorboard. """
    def __init__(self, **kwargs):
        if LOGGER_CHECK["tensorboard"] is False:
            raise ImportError("tensorboard is not installed")
        self.init_logger(**kwargs)

    def init_logger(self, **kwargs):
        self.logger = SummaryWriter(**kwargs)

    def log_metric(self, metric_dict, step):
        for k, v in metric_dict.items():
            self.logger.add_scalar(k, v, step)