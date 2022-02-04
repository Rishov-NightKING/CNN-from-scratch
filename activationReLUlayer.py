import numpy as np
from utils import relu_derivative


class ActivationReLuLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)

        return self.output

    def backward(self, output_gradient, learning_rate):
        return np.dot(output_gradient, relu_derivative(self.input))
