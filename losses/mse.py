from .loss import Loss
from numpy import sum

class MSE(Loss):
    def __init__(self):
        self._grad = 0

    def __call__(self, y_pred, y_true):
        self._grad = (y_pred - y_true) / y_pred.shape[0]
        return ((y_pred - y_true)**2).mean(axis=1) / 2

    def backward(self):
        return self._grad
