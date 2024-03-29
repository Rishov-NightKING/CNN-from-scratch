import numpy as np
from utils import generate_true_value_in_max_value_position


class MaxPoolingLayer:
    # number of output channel = filter counts
    def __init__(self, input_shape, filter_dimension: int, stride: int):
        self.input = None

        self.filter_dimension = filter_dimension
        self.stride = stride

        number_of_input_channel, input_height, input_width = input_shape  # (d, h, w)
        self.input_shape = input_shape
        self.number_of_input_channel = number_of_input_channel

        self.output_shape = (self.number_of_input_channel,
                             np.int((input_height - filter_dimension) / stride) + 1,
                             np.int((input_width - filter_dimension) / stride) + 1)

        self.output = np.zeros(self.output_shape)

    def forward(self, input_data):
        self.input = input_data

        _, output_height, output_width = self.output_shape

        for c in range(self.number_of_input_channel):
            for h in range(output_height):
                vertical_start = self.stride * h
                vertical_end = vertical_start + self.filter_dimension
                for w in range(output_width):
                    horizontal_start = self.stride * w
                    horizontal_end = horizontal_start + self.filter_dimension
                    input_slice = self.input[c, vertical_start:vertical_end, horizontal_start:horizontal_end]
                    self.output[c, h, w] = np.max(input_slice)

        return self.output

    def backward(self, output_gradient, learning_rate):
        output_channel, output_height, output_width = output_gradient.shape

        output_gradient_previous = np.zeros(self.input_shape)

        for c in range(output_channel):
            for h in range(output_height):
                vertical_start = self.stride * h
                vertical_end = vertical_start + self.filter_dimension
                for w in range(output_width):
                    horizontal_start = self.stride * w
                    horizontal_end = horizontal_start + self.filter_dimension
                    input_slice = self.input[c, vertical_start:vertical_end, horizontal_start:horizontal_end]
                    new_window = generate_true_value_in_max_value_position(input_slice)
                    output_gradient_previous[c, vertical_start:vertical_end,
                                        horizontal_start:horizontal_end] += new_window * output_gradient[c, h, w]

        return output_gradient_previous
