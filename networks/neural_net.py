import numpy as np
from activations import *
from losses import MSE
from optimizers import MomentumSGD


class NeuralNet:
    activation_mapping = {
        'arctan': Arctan,
        'linear': Linear,
        'relu': ReLu,
        'sigmoid': Sigmoid,
        'softplus': SoftPlus,
        'tanh': Tanh
    }

    def __init__(self, inputs, hidden, outputs, activations, loss, seed=None):
        assert(len(activations) == len(hidden) + 1)
        ns = [inputs, *hidden, outputs]
        self._seed = seed
        if self._seed is not None:
            np.random.seed(self._seed)

        self._activations = [Linear()] + [self.activation_mapping[a]() for a in activations]
        self._weights = [np.random.normal(size=(ns[i], ns[i+1])) for i in range(len(ns) - 1)]
        self._biases = [np.random.normal(size=ns[i]) for i in range(1, len(ns))]
        self._optimizer = MomentumSGD(0.01, 0.9, self._weights, self._biases)
        self._loss = loss

    def fit(self, x, y):
        self.train()
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        w_grads, b_grads = self.backward()
        self._optimizer.update(w_grads, b_grads)
        return loss

    def predict(self, x):
        self.eval()
        y_pred = self.forward(x)
        return y_pred

    def forward(self, x):
        if len(x.shape) == 1: 
            x = np.expand_dims(x, axis=0)

        x = self._activations[0](x)
        for w, b, a in zip(self._weights, self._biases, self._activations[1:]):
            x = np.matmul(x, w) + b
            x = a(x)
        return x

    def backward(self):
        dz = self._loss.backward() * self._activations[-1].backward()
        w_grads = []
        b_grads = []
        for i in range(len(self._weights)-1, -1, -1):
            dW = np.matmul(np.transpose(self._activations[i].output), dz)
            db = np.sum(dz, axis=0, keepdims=True)
            dz = np.matmul(dz, np.transpose(self._weights[i]))
            dz = dz * self._activations[i].backward()
            w_grads.append(dW)
            b_grads.append(db.squeeze())
        return w_grads[::-1], b_grads[::-1]

    def train(self):
        for a in self._activations: a.train()
        if self._optimizer is not None:
            self._optimizer.train()

    def eval(self):
        for a in self._activations: a.eval()
        if self._optimizer is not None:
            self._optimizer.eval()

    def __call__(self, x):
        return self.forward(x)
