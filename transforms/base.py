
class Transform(object):
    def __init__(self, **kwargs):
        self.is_pass_required = False

    def forward(self, x, **kwargs):
        return x

    def backward(self, x, **kwargs):
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)