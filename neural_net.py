import numpy as np
from activations import Tanh, Sigmoid, Linear
from losses import MSE
from optimizer import MomentumSGD

class NeuralNet:
    activation_mapping = {
        'sigmoid': Sigmoid,
        'tanh': Tanh
    }

    def __init__(self, inputs, hidden, outputs, activations, loss, seed=None):
        ns = [inputs, *hidden, outputs]
        self._seed = seed
        if self._seed is not None:
            np.random.seed(self._seed)

        self._activations = [Linear()] + [self.activation_mapping[a]() for a in activations]
        self._weights = [np.random.normal(size=(ns[i], ns[i+1])) for i in range(len(ns) - 1)]
        self._biases = [np.random.normal(size=ns[i]) for i in range(1, len(ns))]
        self._optimizer = MomentumSGD(0.01, 0.9, self._weights, self._biases)
        self._loss = loss

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

    def fit(self, x, y):
        self.train()
        y_pred = self.forward(x)
        self._loss(y_pred, y)
        w_grads, b_grads = self.backward()
        old = self._weights[-1]
        self._optimizer.update(w_grads, b_grads)
        new = self._weights[-1]

    def train(self):
        for a in self._activations: a.train()
        if self._optimizer is not None:
            self._optimizer.train()

    def eval(self):
        for a in self._activations: a.eval()
        if self._optimizer is not None:
            self._optimizer.eval()

    def __repr__(self):
        return  f'Weights: {"/".join([str(w.shape) for w in self._weights])}\n\
            Biases: {"/".join([str(b.shape) for b in self._biases])}'


if __name__ == '__main__':
    act = ['sigmoid'] * 2
    loss = MSE()
    nn = NeuralNet(inputs=3, hidden=[4], outputs=2, activations=act, loss=loss,
                    seed=20)
    x = np.ones(shape=(3,))
    y = np.ones(shape=(1,2))
    nn.train()
    nn.fit(x, y)
    # nn.backward()