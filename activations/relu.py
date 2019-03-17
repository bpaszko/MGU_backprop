from .activation import Activation

import logging


class ReLu(Activation):
    def __init__(self):
        super().__init__()

    @property
    def output(self):
        return self._output

    def __call__(self, x):
        output = x if x >= 0 else 0
        if self._train:
            self._input = x
            self._output = output
        return output

    def backward(self):
        if self._train:
            return 1 if self._input >= 0 else 0
        else:
            logging.log(logging.WARN, 'Differentiating with train=False')
            return 0
