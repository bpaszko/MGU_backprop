from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def __call__(self, y_pred, y_true):
        pass
