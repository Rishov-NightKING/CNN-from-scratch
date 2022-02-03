import numpy as np
from utils import flatten_matrix


class FullyConnectedLayer:
    def __init__(self, output_dimension):
        self.input = None
        self.output = None

        self.input_dimension = None
        self.output_dimension = output_dimension

        self.weights = None
        self.bias = np.random.randn(output_dimension, 1)
        # self.bias = np.full((output_dimension, 1), 20, dtype='float64')

    def forward(self, input_data):
        self.input = flatten_matrix(input_data)
        self.input_dimension, _ = self.input.shape
        self.weights = np.random.randn(self.output_dimension, self.input_dimension)
        # self.weights = np.full((self.output_dimension, self.input_dimension), 15, dtype='float64')
        self.output = np.matmul(self.weights, self.input) + self.bias

        return self.output
