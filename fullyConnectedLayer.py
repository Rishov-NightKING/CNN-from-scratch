import numpy as np
from utils import flatten_matrix


class FullyConnectedLayer:
    def __init__(self, input_dimension, output_dimension):
        self.input = None
        self.output = None

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.weights = np.random.randn(self.output_dimension, self.input_dimension) / self.input_dimension
        # changed here 100 diye divide / input dimension diye
        self.bias = np.random.randn(output_dimension, 1)
        # self.bias = np.full((output_dimension, 1), 20, dtype='float64')

    def forward(self, input_data):
        self.input = input_data

        # self.weights = np.full((self.output_dimension, self.input_dimension), 15, dtype='float64')
        self.output = np.matmul(self.weights, self.input) + self.bias

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.matmul(np.transpose(self.weights), output_gradient)
        weights_gradient = np.matmul(output_gradient, np.transpose(self.input))
        bias_gradient = output_gradient

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
