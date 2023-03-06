import typing
import torch

from models import create_model
from models.optimizers import create_optimizer

class UncertaintyEstimator(object):
    def init_estimator(self, **kwargs):
        raise NotImplementedError

    def init_predictor(self, **kwargs):
        raise NotImplementedError
    
    def train_estimator(self, **kwargs):
        raise NotImplementedError

    def test_estimator(self, **kwargs):
        raise NotImplementedError
    
    def train_predictor(self, **kwargs):
        raise NotImplementedError
    
    def test_predictor(self, **kwargs):
        raise NotImplementedError
    
    def _build_network(self, network_build_config, network_name):
        return create_model(network_build_config.get(network_name, None))

    def _build_optimizer(self, optimizer_build_config, params):
        return create_optimizer(params, optimizer_build_config)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'