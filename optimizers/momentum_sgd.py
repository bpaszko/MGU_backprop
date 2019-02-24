from .optimizer import Optimizer

import numpy as np


class MomentumSGD(Optimizer):
    def __init__(self, lr, momentum, weights, biases):
        super().__init__()
        self._lr = lr
        self._momentum = momentum
        self._weights = weights
        self._biases = biases
        self._w_grads = []
        self._b_grads = []

    def update(self, new_w_grads, new_b_grads):
        for i in range(len(self._weights)):
            w_update = self._w_grads[i] * self._momentum + new_w_grads[i] * (1-self._momentum)
            b_update = self._b_grads[i] * self._momentum + new_b_grads[i] * (1-self._momentum)
            self._weights[i] = self._weights[i] - self._lr * w_update
            self._biases[i] = self._biases[i] - self._lr * b_update

    def train(self):
        if not self._w_grads:
            self._w_grads = [np.zeros_like(w) for w in self._weights]
            self._b_grads = [np.zeros_like(b) for b in self._biases]

    def eval(self):
        self._w_grads = []
        self._b_grads = []
