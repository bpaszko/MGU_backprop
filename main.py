from neural_net import NeuralNet
from losses import MSE
import pandas as pd
import numpy as np
from tqdm import tqdm


class Provider:
    def __init__(self, df, shuffle=False):
        self._df = df
        self._pos = 0 

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self):
        if self._pos == 0:
            self._pos = 1
            return (self._df[['x', 'y']].values, 
                np.expand_dims(self._df['cls'].apply(lambda x: x - 1).values, axis=1))
        else:
            raise StopIteration

def train(network, epochs, provider_train, provider_test):
    for epoch in tqdm(range(epochs)):
        network.train()
        for iteration, (data, labels) in enumerate(provider_train):
            # print(data.shape)
            # print(labels.shape)
            network.fit(data, labels)

        network.eval()
        if epoch == epochs - 1:
            for (data, labels) in provider_test:
                preds = network.forward(data)
                print(labels[:10])
                print(preds[:10])
            
            


if __name__ == '__main__':
    train_df = pd.read_csv('data/Classification/data.simple.train.500.csv')
    test_df = pd.read_csv('data/Classification/data.simple.test.100.csv')
    p_train = Provider(train_df)
    p_test = Provider(test_df)

    act = ['sigmoid'] * 3
    loss = MSE()
    nn = NeuralNet(inputs=2, hidden=[100, 50], outputs=1, activations=act, loss=loss,
                    seed=20)
    train(nn, 500, p_train, p_test)