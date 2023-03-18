
import torch
class Transform(torch.nn.Module): # Transforms are simple layers with no parameters. Allow pytorch models to be used as transforms.
    def __init__(self, **kwargs):
        super(Transform, self).__init__()
        self.is_pass_required = False

    def forward(self, x, **kwargs):
        return x

    def backward(self, x, **kwargs):
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
class PytorchWrapper(Transform):
    """ A simple wrapper for pytorch models."""
    def __init__(self, transform, **kwargs):
        super(PytorchWrapper, self).__init__()
        self.transform = transform

    def forward(self, x, **kwargs):
        return self.transform(x, **kwargs)
    
    def backward(self, x, **kwargs):
        return x
    
    def __repr__(self):
        return self.transform.__repr__()
    
