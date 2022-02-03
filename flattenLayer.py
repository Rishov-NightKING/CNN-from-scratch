from utils import flatten_matrix


class FlattenLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        self.output = flatten_matrix(input_data)

        return self.output

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)