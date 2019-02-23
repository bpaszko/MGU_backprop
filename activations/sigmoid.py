from .activation import Activation

import numpy as np
import logging


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    @property
    def output(self):
        return self._output

    def __call__(self, x):
        output = 1 / (1 + np.exp(-x))
        if self._train:
            self._input = x
            self._output = output
        return output

    def backward(self):
        if self._train:
            return self._output * (1 - self._output)
        else:
            logging.log(logging.WARN, 'Differentiating with train=False')
            return 0