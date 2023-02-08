import numpy as np
class ToyFunc(object):
    def __init__(self, func, repr_str):
        self.func = func
        self.repr = repr_str

    def __call__(self, x):
        return self.func(x)

    def __repr__(self) -> str:
        return self.repr_str

def toy_function_sin():
    return ToyFunc(np.sin, repr_str = "sin(x)")

def toy_function_cos():
    return ToyFunc(np.cos, repr_str = "cos(x)")

def toy_func_mixed():
    def mixed(x):
        return np.sin(x) - np.cos(x/2) + np.sin(x/4) - np.sin(x/8)
    return ToyFunc(mixed, repr_str = "sin(x) + cos(x/2) + sin(x/4) + 3*sin(x/8)")
