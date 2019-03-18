from networks.neural_net import NeuralNet
from losses import MSE
from providers import ClassifierProvider, RegressionProvider
from json_parser import JsonParser
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class VisualizationProvider:
    def __init__(self, df, nn, assignment_type):
        assert assignment_type in {"classification", "regression"}, "Wrong assignment type"
        self._assignment_type = assignment_type
        self._df = df
        self._nn = nn

    def plot(self):
        if self._assignment_type == "regression":
            x = self._df[["x"]].values
            y_true = self._df[["y"]].values
            y_pred = nn.forward(x)
            plt.plot(x, y_true, color='blue', label="true")
            plt.plot(x, y_pred, 'r--', label="predicted")
            plt.legend()
            plt.show()
        elif self._assignment_type == "classification":
            x = self._df[["x", "y"]].values
            y_true = self._df[["cls"]].values
            y_pred = self._nn.predict(x)
            plt.subplot(121)
            plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y_true), alpha=0.2)
            plt.title("True")
            plt.subplot(122)
            plt.scatter(x[:, 0], x[:, 1], c=np.argmax(y_pred, axis=1))
            plt.title("Predicted")
            plt.show()


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
            train_loss_per_epoch += loss
        train_loss_per_epoch /= len(provider_train)
        summary.train_losses.append((epoch, train_loss_per_epoch))

        network.eval()
        test_loss = 0
        if epoch % test_epochs == 0:
            for (data, labels) in provider_test:
                preds = network(data)
                loss = network._loss(preds, labels)
                test_loss += loss
            test_loss /= len(provider_test)
            summary.test_losses.append((epoch, test_loss))
    return summary
            

if __name__ == '__main__':
    json_parser = JsonParser("architecture.json")
    json_parser.parse_json()
    train_df = pd.read_csv(json_parser.input_train_file_path)
    test_df = pd.read_csv(json_parser.input_test_file_path)

    type_of_assigment = json_parser.type
    p_train = None
    p_test = None
    output_layer = None
    if type_of_assigment == "regression":
        p_train = RegressionProvider(train_df, batch_size=json_parser.batch_size)
        p_test = RegressionProvider(test_df, batch_size=json_parser.batch_size)
        output_layer = "linear"
    elif type_of_assigment == "classification":
        p_train = ClassifierProvider(train_df, batch_size=json_parser.batch_size)
        p_test = ClassifierProvider(test_df, batch_size=json_parser.batch_size)
        output_layer = "sigmoid"
    hidden = json_parser.layers_size
    act = json_parser.layers_activations
    act.append(output_layer)
    seed = json_parser.seed
    loss = MSE()
    number_of_iterations = json_parser.number_of_iterations

    nn = NeuralNet(inputs=p_train.number_of_inputs, hidden=hidden, outputs=p_train.number_of_outputs,
                   activations=act, loss=loss, seed=seed)

    summary = train(nn, number_of_iterations, p_train, p_test)
    summary.show()
    visualization_provider = VisualizationProvider(test_df, nn, type_of_assigment)
    visualization_provider.plot()

    # TODO should allow to show weights during training (save to file on request?)
