from .activation import Activation

import numpy as np
import logging


class ReLu(Activation):
    def __init__(self):
        super().__init__()

    @property
    def output(self):
        return self._output

    def __call__(self, x):
        output = np.maximum(x, 0)
        if self._train:
            self._input = x
            self._output = output
        return output

    def backward(self):
        if self._train:
            return np.ones_like(self._input) * (self._input > 0)
        else:
            logging.log(logging.WARN, 'Differentiating with train=False')
            return 0
