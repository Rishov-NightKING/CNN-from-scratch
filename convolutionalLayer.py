import numpy as np
from utils import cross_correlation2d, pad_input_2d_mat


class ConvolutionalLayer:
    # number of output channel = filter counts
    def __init__(self, input_shape, number_of_output_channel: int, filter_dimension: int,
                 stride: int, padding: int):
        self.input = None
        self.output = None
        self.number_of_output_channel = number_of_output_channel
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.padding = padding

        number_of_input_channel, input_height, input_width = input_shape  # (d, h, w)
        self.input_shape = input_shape
        self.number_of_input_channel = number_of_input_channel

        self.output_shape = (number_of_output_channel,
                             np.int((input_height - filter_dimension + 2 * padding) / stride) + 1,
                             np.int((input_width - filter_dimension + 2 * padding) / stride) + 1)

        self.biases = np.random.randn(number_of_output_channel,
                                      np.int((input_height - filter_dimension + 2 * padding) / stride) + 1,
                                      np.int((input_width - filter_dimension + 2 * padding) / stride) + 1)
        self.filters_weights = np.random.randn(number_of_output_channel, number_of_input_channel, filter_dimension,
                                               filter_dimension)
        self.filters_weights *= (np.sqrt(2) / np.prod(self.filters_weights.shape))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.biases)

        for i in range(self.number_of_output_channel):
            for j in range(self.number_of_input_channel):
                self.output[i] += cross_correlation2d(self.input[j], self.filters_weights[i, j], stride=self.stride,
                                                      padding=self.padding, result_shape=self.output_shape[1:])
        return self.output

    def backward(self, output_gradient, learning_rate):  # output_gradient = dE/dY(i)
        filters_weight_gradient = np.zeros((self.number_of_output_channel, self.number_of_input_channel,
                                            self.filter_dimension, self.filter_dimension))  # dE/dK(i,j)
        bias_gradient = output_gradient  # dE/dY(i)

        for i in range(self.number_of_output_channel):
            for j in range(self.number_of_input_channel):
                filters_weight_gradient[i, j] = cross_correlation2d(self.input[j], output_gradient[i],
                                                                    stride=self.stride, padding=self.padding,
                                                                    result_shape=(
                                                                        self.filter_dimension,
                                                                        self.filter_dimension))

        output_channel, output_height, output_width = output_gradient.shape

        input_gradient_pad = np.zeros(self.input_shape)
        padded_2d_arr = []
        for c in range(self.number_of_input_channel):
            padded_2d_arr.append(pad_input_2d_mat(input_gradient_pad[c], pad=self.padding))
        input_gradient_pad = np.array(padded_2d_arr)

        for c in range(output_channel):
            for h in range(output_height):
                vertical_start = self.stride * h
                vertical_end = vertical_start + self.filter_dimension
                for w in range(output_width):
                    horizontal_start = self.stride * w
                    horizontal_end = horizontal_start + self.filter_dimension
                    input_gradient_pad[:, vertical_start:vertical_end,
                    horizontal_start:horizontal_end] += self.filters_weights[c] * output_gradient[c, h, w]

        self.filters_weights -= learning_rate * filters_weight_gradient
        self.biases -= learning_rate * bias_gradient

        if self.padding > 0:
            return input_gradient_pad[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return input_gradient_pad
