import json
import os.path
from networks.neural_net import NeuralNet


class JsonParser:
    jsonAttributes = {
        "learning_rate",
        "momentum",
        "number_of_iterations",
        "seed",
        "hidden_layers",
        "type",
        "train_df",
        "test_df",
        "batch_size"
    }

    def __init__(self, json_path):
        self._json_path = json_path
        self._json_data = None
        with open(self._json_path, "r") as json_file:
            self._json_data = json.loads(json_file.read())
            assert self.jsonAttributes == self._json_data.keys(), "Wrong JSON format"
        self.learning_rate = None
        self.momentum = None
        self.number_of_iterations = None
        self.seed = None
        self.input_test_file_path = None
        self.input_train_file_path = None
        self.type = None
        self.output_layer_activation = None
        self.batch_size = None
        self.layers_size = []
        self.layers_activations = []

    def parse_json(self):
        data = self._json_data
        self.learning_rate = data["learning_rate"]
        assert 0 < self.learning_rate < 1, "Wrong learning rate value"
        self.momentum = data["momentum"]
        assert 0 <= self.momentum <= 1, "Wrong momentum value"
        self.number_of_iterations = data["number_of_iterations"]
        assert self.number_of_iterations > 0, "Wrong number of iterations value"
        self.seed = data["seed"]
        self.input_test_file_path = data["test_df"]
        assert os.path.isfile(self.input_test_file_path), "Test file doesn't exist"
        self.input_train_file_path = data["train_df"]
        assert os.path.isfile(self.input_train_file_path), "Train file doesn't exist"
        self.type = data["type"]
        assert self.type in ["classification", "regression"], "Wrong assignment type"
        self.batch_size = int(data["batch_size"])
        assert int(self.batch_size) > 0, "Wrong batch size value"
        for layer in data["hidden_layers"]:
            size = int(layer["neurons"])
            assert size > 0, "Wrong layer size"
            assert layer["activation"] in NeuralNet.activation_mapping.keys(), "Wrong activation type"
            self.layers_size.append(size)
            self.layers_activations.append(layer["activation"])
