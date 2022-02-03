import numpy as np


class ActivationReLuLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)

        return self.output
