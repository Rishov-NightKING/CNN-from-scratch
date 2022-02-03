import numpy as np


class MaxPoolingLayer:
    # number of output channel = filter counts
    def __init__(self, input_shape: (int, int, int), filter_dimension: int, stride: int):
        self.input = None
        # self.output = None

        self.filter_dimension = filter_dimension
        self.stride = stride

        number_of_input_channel, input_height, input_width = input_shape  # (d, h, w)
        self.input_shape = input_shape
        self.number_of_input_channel = number_of_input_channel

        if stride != 0:
            self.output_shape = (self.number_of_input_channel,
                                 np.int((input_height - filter_dimension) / stride) + 1,
                                 np.int((input_width - filter_dimension) / stride) + 1)
        else:
            self.output_shape = (self.number_of_input_channel,
                                 np.int(input_height - filter_dimension) + 1,
                                 np.int(input_width - filter_dimension) + 1)
        self.output = np.zeros(self.output_shape)

    def forward(self, input_data):
        self.input = input_data

        _, output_height, output_width = self.output_shape

        for c in range(self.number_of_input_channel):
            for h in range(output_height):  # loop on the vertical axis of the output volume
                # Find the vertical start and end of the current "slice" (≈2 lines)
                vertical_start = self.stride * h
                vertical_end = vertical_start + self.filter_dimension

                for w in range(output_width):  # loop on the horizontal axis of the output volume
                    # Find the vertical start and end of the current "slice" (≈2 lines)
                    horizontal_start = self.stride * w
                    horizontal_end = horizontal_start + self.filter_dimension
                    input_slice = self.input[c, vertical_start:vertical_end, horizontal_start:horizontal_end]
                    self.output[c, h, w] = np.max(input_slice)

        return self.output
