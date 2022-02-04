import numpy as np

from cnn import ConvolutionalNeuralNetwork
from convolutionalLayer import ConvolutionalLayer
from maxPoolingLayer import MaxPoolingLayer
from fullyConnectedLayer import FullyConnectedLayer
from activationReLUlayer import ActivationReLuLayer
from flattenLayer import FlattenLayer
from softMaxLayer import SoftMaxLayer
from utils import file_read, mnist_dataset_load_and_preprocess, cifar10_dataset_load_and_preprocess

if __name__ == "__main__":
    np.random.seed(1)
    x_train, y_train, x_test, y_test = mnist_dataset_load_and_preprocess()
    # x_train, y_train, x_test, y_test = cifar10_dataset_load_and_preprocess()

    inputs_given_in_file = file_read(file_path='input.txt')
    model = ConvolutionalNeuralNetwork()

    layer_shape = x_train.shape[1:]

    for layer_info in inputs_given_in_file:
        layer_name = layer_info[0]
        if layer_name == 'Conv':
            model.add_layer(ConvolutionalLayer(input_shape=layer_shape, number_of_output_channel=layer_info[1],
                                               filter_dimension=layer_info[2], stride=layer_info[3],
                                               padding=layer_info[4]))
            layer_shape = model.layers[-1].output_shape

        elif layer_name == 'ReLU':
            model.add_layer(ActivationReLuLayer())
            # layer_shape unchanged

        elif layer_name == 'Pool':
            model.add_layer(MaxPoolingLayer(input_shape=layer_shape, filter_dimension=layer_info[1],
                                            stride=layer_info[2]))
            layer_shape = model.layers[-1].output_shape

        elif layer_name == 'FC':
            input_dim = np.prod(layer_shape)
            model.add_layer(FlattenLayer())
            model.add_layer(FullyConnectedLayer(input_dimension=input_dim, output_dimension=layer_info[1]))
            layer_shape = (input_dim, 1)

        elif layer_name == 'Softmax':
            model.add_layer(SoftMaxLayer())
            # layer shape unchanged

    model.train_model(x_train, y_train, learning_rate=0.005, number_of_epochs=2)

    model.predict(x_test, y_test)

    # dataset = np.random.randn(1, 28, 28)

    # out1 = ConvolutionalLayer(input_shape=(1, 28, 28), number_of_output_channel=6, filter_dimension=5, stride=1,
    #                           padding=2).forward(input)
    # out2 = ActivationReLuLayer().forward(out1)
    # out3 = MaxPoolingLayer(input_shape=out2.shape, filter_dimension=2, stride=2).forward(out2)
    #
    # out4 = ConvolutionalLayer(input_shape=out3.shape, number_of_output_channel=12, filter_dimension=5, stride=1,
    #                           padding=0).forward(out3)
    # out5 = ActivationReLuLayer().forward(out4)
    # out6 = MaxPoolingLayer(input_shape=out5.shape, filter_dimension=2, stride=2).forward(out5)
    #
    # out7 = ConvolutionalLayer(input_shape=out6.shape, number_of_output_channel=100, filter_dimension=5, stride=1,
    #                           padding=0).forward(out6)
    # out8 = ActivationReLuLayer().forward(out7)
    #
    # # print(out8)
    #
    # out9 = FlattenLayer().forward(out8)
    #
    # out10 = FullyConnectedLayer(output_dimension=10).forward(out9)
    #
    # out11 = SoftMaxLayer().forward(out10)
    # print(out11)
