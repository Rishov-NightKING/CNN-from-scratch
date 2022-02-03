import numpy as np
from keras.utils import np_utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10


def file_read(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    lines = [[int(x) if idx != 0 else x for idx, x in enumerate(line)] for line in lines]

    '''
    *********************************** INPUT FORMAT *********************************************
    Conv: the number of output channels, filter dimension, stride, padding.
    ReLU: Activation layer.
    Pool: filter dimension, stride.
    FC: output dimension.
    Flattening layer: it will convert a (series of) convolutional filter maps to a column vector.
    Softmax: it will convert final layer projections to normalized probabilities.
    '''

    return lines


def pad_input_2d_mat(X, pad: int, padding_value: int = 0):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C) # m is omitted for now
    """
    X_pad = np.pad(X, pad_width=((pad, pad), (pad, pad)),
                   mode='constant', constant_values=(padding_value, padding_value))
    # X_pad = np.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)))

    return X_pad


def cross_correlation2d(input_mat, filter_mat, stride, padding, result_shape):
    filter_dimension = filter_mat.shape[0]
    result_height, result_width = result_shape

    result = np.zeros((result_height, result_width), dtype='float64')

    if padding > 0:
        pad_input = pad_input_2d_mat(input_mat, pad=padding)
    else:
        pad_input = input_mat

    for h in range(result_height):  # loop over vertical axis of the output volume
        # Find the vertical start and end of the current "slice" (≈2 lines)
        vertical_start = stride * h
        vertical_end = vertical_start + filter_dimension

        for w in range(result_width):  # loop over horizontal axis of the output volume
            # Find the horizontal start and end of the current "slice" (≈2 lines)
            horizontal_start = stride * w
            horizontal_end = horizontal_start + filter_dimension

            input_slice = pad_input[vertical_start:vertical_end, horizontal_start:horizontal_end]
            result[h, w] = np.sum(np.multiply(input_slice, filter_mat))
            # print(result[h, w])

    return result


def convolution2d(input_mat, filter_mat, stride, padding, result_shape):
    # Flip the filter
    rotated_filter = np.flipud(np.fliplr(filter_mat))

    return cross_correlation2d(input_mat, rotated_filter, stride, padding, result_shape)


def flatten_matrix(input_data):
    return input_data.flatten().reshape((-1, 1))


def softmax(input_data):
    exponent = np.exp(input_data)
    exponent_sum = np.sum(exponent)
    return exponent / exponent_sum


def preprocess_mnist_data(x, y):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float64") / 255
    # y = np_utils.to_categorical(y) # confusion
    y = y.reshape(-1, 1)

    return x, y


def preprocess_cifar10_data(x, y): # dimension (60k, 32, 32, 3)
    x = np.transpose(x, (0, 3, 1, 2)) # changed dimension (60k, 3, 32, 32)
    x = x.astype("float64") / 255
    # y = np_utils.to_categorical(y) # confusion

    return x, y


def mnist_dataset_load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist_data(x_train, y_train)
    x_test, y_test = preprocess_mnist_data(x_test, y_test)

    return x_train, y_train, x_test, y_test


def cifar10_dataset_load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # dimension (60k, 32, 32, 3)
    x_train, y_train = preprocess_cifar10_data(x_train, y_train)
    x_test, y_test = preprocess_cifar10_data(x_test, y_test)

    return x_train, y_train, x_test, y_test
