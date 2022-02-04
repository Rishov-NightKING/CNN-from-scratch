import numpy as np
from utils import softmax


class SoftMaxLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = softmax(input_data)

        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient
