import numpy as np
import time
import threading

from elapsedTime import ElapsedTimeThread
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
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist_dataset_load_and_preprocess()
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
            layer_shape = model.forward_layers[-1].output_shape

        elif layer_name == 'ReLU':
            model.add_layer(ActivationReLuLayer())
            # layer_shape unchanged

        elif layer_name == 'Pool':
            model.add_layer(MaxPoolingLayer(input_shape=layer_shape, filter_dimension=layer_info[1],
                                            stride=layer_info[2]))
            layer_shape = model.forward_layers[-1].output_shape

        elif layer_name == 'FC':
            input_dim = np.prod(layer_shape)
            model.add_layer(FlattenLayer())
            model.add_layer(FullyConnectedLayer(input_dimension=input_dim, output_dimension=layer_info[1]))
            layer_shape = (input_dim, 1)

        elif layer_name == 'Softmax':
            model.add_layer(SoftMaxLayer())
            # layer shape unchanged

    start = time.time()
    thread = ElapsedTimeThread()
    thread.start()

    model.train_model(x_train[0:100], y_train[0:100], learning_rate=0.005, number_of_epochs=2)

    thread.stop()
    thread.join()
    print("\nTraining finished in {:.3f} seconds".format(time.time() - start))

