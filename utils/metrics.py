import torch
import torch.nn as nn 


def rmse(y_pred, y_true, **kwargs):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

def mae(y_pred, y_true, **kwargs):
    return torch.mean(torch.abs(y_pred - y_true))

def mape(y_pred, y_true, **kwargs):
    return torch.mean(torch.abs((y_pred - y_true) / y_true))

def mse(y_pred, y_true, **kwargs):
    return torch.mean((y_pred - y_true) ** 2)

def gm(y_pred, y_true, **kwargs):
    """Calculate the geometric mean of the error"""
    return torch.exp(torch.mean(torch.log(torch.abs(y_pred - y_true))))

def nll(y_pred, y_true, **kwargs):
    return -torch.mean(y_pred.log_prob(y_true))

def nll_mse(y_pred, y_true, **kwargs):
    return -torch.mean(y_pred.log_prob(y_true)) + torch.mean((y_pred.mean - y_true) ** 2)

def nll_ensemble(y_pred, y_true, sigma_pred, **kwargs):
    return torch.mean((0.5*torch.log(sigma_pred) + 0.5*(torch.square(y_true - y_pred))/sigma_pred)) 

def weighted_mse(y_pred, y_true, weights, **kwargs):
    return torch.mean(weights * (y_pred - y_true) ** 2)

def weighted_rmse(y_pred, y_true, weights, **kwargs):
    return torch.sqrt(torch.mean(weights * (y_pred - y_true) ** 2))

def weighted_mae(y_pred, y_true, weights, **kwargs):
    return torch.mean(weights * torch.abs(y_pred - y_true))

def pearson_correlation(y_pred, y_true, **kwargs):
    return torch.mean((y_pred - torch.mean(y_pred)) * (y_true - torch.mean(y_true))) / torch.sqrt(
        torch.mean((y_pred - torch.mean(y_pred)) ** 2) * torch.mean((y_true - torch.mean(y_true)) ** 2))


class MetricList:
    def __init__(self, metric_list = []):
        self.metric_list = metric_list
        self.metrics = {}
        self.reset()

    def reset(self):
        for metric in self.metric_list:
            self.metrics[str(metric)] = []

    def get_metrics(self, add_prefix = None):
        if add_prefix is not None:
            return {add_prefix + "_" + str(metric): self.metrics[str(metric)] for metric in self.metric_list}
        return self.metrics
    
    def get_metric(self, metric):
        return self.metrics[metric]
    
    @torch.no_grad()
    def forward(self, y_pred, y_true, **kwargs):
        for metric in self.metric_list:
            self.metrics[str(metric)] = metric(y_pred, y_true, **kwargs)

    def __repr__(self) -> str:
        base = "MetricList("
        for metric in self.metric_list:
            base += str(metric) + ", "
        base = base[:-2] + ")"
        return base
        
    
class Metric:
    def __init__(self, metric_fn, metric_name = None):
        self.metric_fn = metric_fn
        self.metric_name = metric_name if metric_name is not None else metric_fn.__name__

    def __call__(self, *args, **kwargs):
        return self.metric_fn(*args, **kwargs)
    
    def __str__(self):
        return self.metric_name
    
    def __repr__(self) -> str:
        self.__str__()





