import typing
import torch
import abc



class UncertaintyEstimator(object):
    def init_estimator(self, **kwargs):
        raise NotImplementedError

    def train_estimator(self, **kwargs):
        raise NotImplementedError

    def test_estimator(self, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'