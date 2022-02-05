import numpy as np
from utils import cross_correlation2d, pad_input_2d_mat


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
        self.input_shape = input_shape
        self.number_of_input_channel = number_of_input_channel

        self.output_shape = (number_of_output_channel,
                             np.int((input_height - filter_dimension + 2 * padding) / stride) + 1,
                             np.int((input_width - filter_dimension + 2 * padding) / stride) + 1)

        self.filters_shape = (number_of_output_channel, number_of_input_channel, filter_dimension, filter_dimension)

        self.biases = np.random.randn(number_of_output_channel,
                                      np.int((input_height - filter_dimension + 2 * padding) / stride) + 1,
                                      np.int((input_width - filter_dimension + 2 * padding) / stride) + 1)
        self.filters_weights = np.random.randn(number_of_output_channel, number_of_input_channel, filter_dimension,
                                               filter_dimension)
        self.filters_weights *= (np.sqrt(2)/np.prod(self.filters_weights.shape))
        # output = WX + b so biases are stored early here to add later
        # self.filters_weights = np.full(self.filters_shape, 0.5) / 100
        # self.biases = np.full(self.output_shape, 0.1)

    def forward(self, input_data):
        self.input = input_data
        self.output = np.copy(self.biases)

        for i in range(self.number_of_output_channel):
            for j in range(self.number_of_input_channel):
                self.output[i] += cross_correlation2d(self.input[j], self.filters_weights[i, j], stride=self.stride,
                                                      padding=self.padding, result_shape=self.output_shape[1:])
        return self.output

    def backward(self, output_gradient, learning_rate):  # output_gradient = dE/dY(i)
        filters_gradient = np.zeros((self.number_of_output_channel, self.number_of_input_channel,
                                     self.filter_dimension, self.filter_dimension))  # dE/dK(i,j)
        input_gradient = np.zeros(self.input_shape)  # dE *(full) K(i,j)
        bias_gradient = output_gradient  # dE/dY(i)

        for i in range(self.number_of_output_channel):
            for j in range(self.number_of_input_channel):
                filters_gradient[i, j] = cross_correlation2d(self.input[j], output_gradient[i],
                                                             stride=self.stride, padding=self.padding,
                                                             result_shape=(
                                                                 self.filter_dimension,
                                                                 self.filter_dimension))  # confusion

        # print("conv: ", filters_gradient.mean())

        output_channel, output_height, output_width = output_gradient.shape

        padded_input = np.zeros(self.input_shape)
        dA_prev_pad = np.zeros(self.input_shape)
        arr1 = []
        arr2 = []
        # if self.padding > 0:
        #     print("hi more jbo")
        for c in range(self.number_of_input_channel):
            arr1.append(pad_input_2d_mat(padded_input[c], pad=self.padding))
            arr2.append(pad_input_2d_mat(dA_prev_pad[c], pad=self.padding))
        padded_input = np.array(arr1)
        dA_prev_pad = np.array(arr2)
        # print("self: ", self.input.shape)
        # print("input jbej: ", padded_input.shape)
        # print("prev ga: ", dA_prev_pad.shape)
        for c in range(output_channel):
            for h in range(output_height):
                vertical_start = self.stride * h
                vertical_end = vertical_start + self.filter_dimension
                for w in range(output_width):
                    horizontal_start = self.stride * w
                    horizontal_end = horizontal_start + self.filter_dimension
                    # a_slice = padded_input[c, vertical_start:vertical_end, horizontal_start:horizontal_end]
                    dA_prev_pad[:, vertical_start:vertical_end,
                    horizontal_start:horizontal_end] += self.filters_weights[c] * output_gradient[c, h, w]

        # print("after: ", dA_prev_pad[:, self.padding:-self.padding, self.padding:-self.padding].shape)
        self.filters_weights -= learning_rate * filters_gradient
        self.biases -= learning_rate * bias_gradient

        # print("conv weight", self.filters_weights.mean())
        # print("conv: ", output_gradient.mean())

        if self.padding > 0:
            return dA_prev_pad[:, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return dA_prev_pad
