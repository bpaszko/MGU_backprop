from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def update(self, new_w_grads, new_b_grads):
        pass
