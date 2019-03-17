from networks.neural_net import NeuralNet
from losses import MSE
from providers import ClassifierProvider, RegressionProvider
from json_parser import JsonParser
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TrainSummary:
    def __init__(self):
        self.train_losses = []
        self.test_losses = []

    def show(self):
        plt.plot([loss[0] for loss in self.train_losses], [loss[1] for loss in self.train_losses],
                  label='Train loss')
        plt.plot([loss[0] for loss in self.test_losses], [loss[1] for loss in self.test_losses],
                  label='Test loss')
        plt.title('Loss plot')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


def train(network, epochs, provider_train, provider_test, test_epochs=5):
    summary = TrainSummary()
    for epoch in tqdm(range(epochs)):
        network.train()
        train_loss_per_epoch = 0
        for iteration, (data, labels) in enumerate(provider_train):
            loss = network.fit(data, labels)
            train_loss_per_epoch += np.sum(loss)
        train_loss_per_epoch /= len(provider_train)
        summary.train_losses.append((epoch, train_loss_per_epoch))

        network.eval()
        test_loss = 0
        if epoch % test_epochs == 0:
            for (data, labels) in provider_test:
                preds = network(data)
                loss = network._loss(preds, labels)
                test_loss += np.sum(loss)
            test_loss /= len(provider_test)
            summary.test_losses.append((epoch, test_loss))
    return summary
            

if __name__ == '__main__':
    json_parser = JsonParser("architecture.json")
    json_parser.parse_json()
    train_df = pd.read_csv(json_parser.input_train_file_path)
    test_df = pd.read_csv(json_parser.input_test_file_path)
    p_train = ClassifierProvider(train_df, batch_size=json_parser.batch_size)
    p_test = ClassifierProvider(test_df, batch_size=json_parser.batch_size)

    hidden = json_parser.layers_size
    act = json_parser.layers_activations
    seed = json_parser.seed
    loss = MSE()
    number_of_iterations = json_parser.number_of_iterations

    nn = NeuralNet(inputs=2, hidden=hidden, outputs=1, activations=act, loss=loss,
                    seed=seed)

    summary = train(nn, number_of_iterations, p_train, p_test)
    summary.show()
