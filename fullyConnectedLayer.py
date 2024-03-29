import numpy as np
from utils import flatten_matrix


class FullyConnectedLayer:
    def __init__(self, input_dimension, output_dimension):
        self.input = None
        self.output = None

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.weights = np.random.randn(self.output_dimension, self.input_dimension)
        self.weights *= (np.sqrt(2) / np.prod(self.weights.shape))
        self.bias = np.random.randn(output_dimension, 1)

    def forward(self, input_data):
        self.input = input_data

        self.output = np.matmul(self.weights, self.input) + self.bias

        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.matmul(np.transpose(self.weights), output_gradient)
        weights_gradient = np.matmul(output_gradient, np.transpose(self.input))
        bias_gradient = output_gradient

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
