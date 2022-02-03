import numpy as np
from utils import cross_correlation2d


class ConvolutionalLayer:
    # number of output channel = filter counts
    def __init__(self, input_shape: (int, int, int), number_of_output_channel: int, filter_dimension: int,
                 stride: int, padding: int):
        self.input = None
        self.output = None
        self.number_of_output_channel = number_of_output_channel
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.padding = padding

        number_of_input_channel, input_height, input_width = input_shape  # (d, h, w)
        # self.input_shape = input_shape
        self.number_of_input_channel = number_of_input_channel

        if stride != 0:
            self.output_shape = (number_of_output_channel,
                                 np.int((input_height - filter_dimension + 2 * padding) / stride) + 1,
                                 np.int((input_width - filter_dimension + 2 * padding) / stride) + 1)
        else:
            self.output_shape = (number_of_output_channel,
                                 np.int(input_height - filter_dimension + 2 * padding) + 1,
                                 np.int(input_width - filter_dimension + 2 * padding) + 1)

        # self.filters_shape = (number_of_output_channel, number_of_input_channel, filter_dimension, filter_dimension)
        # self.filters_weights = np.full(self.filters_shape, 5, dtype='float64')
        self.biases = np.random.randn(*self.output_shape)
        self.filters_weights = np.random.randn(number_of_output_channel, number_of_input_channel, filter_dimension,
                                               filter_dimension)
        self.output = np.copy(self.biases)  # output = WX + b so biases are stored early here to add later

        # self.biases = np.full(self.output_shape, 10, dtype='float64')

    def forward(self, input_data):
        self.input = input_data

        for i in range(self.number_of_output_channel):
            for j in range(self.number_of_input_channel):
                # print(self.filters_weights[i, j].shape)
                self.output[i] += cross_correlation2d(self.input[j], self.filters_weights[i, j], stride=self.stride,
                                                      padding=self.padding, result_shape=self.output_shape)
        return self.output
