
class Transform(object):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        return x

    def backward(self, x, **kwargs):
        return x
