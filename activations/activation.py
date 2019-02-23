from abc import ABC, abstractmethod


class Activation:
    def __init__(self):
        self._train = False
        self._input = None
        self._output = None
    
    def train(self, train=True):
        self._train = train

    def eval(self):
        self.train(False)

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def backward(self):
        pass
