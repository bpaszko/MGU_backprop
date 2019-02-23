from .activation import Activation

import numpy as np
import logging


class Linear(Activation):
    def __init__(self):
        super().__init__()

    @property
    def output(self):
        return self._input

    def __call__(self, x):
        if self._train:
            self._input = x
        return x

    def backward(self):
        if self._train:
            return np.ones_like(self._input)
        else:
            logging.log(logging.WARN, 'Differentiating with train=False')
            return 0