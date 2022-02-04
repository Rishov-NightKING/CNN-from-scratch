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
    exponent = np.exp(input_data-np.max(input_data))
    exponent_sum = np.sum(exponent)
    return exponent / exponent_sum


def generate_true_value_in_max_value_position(specific_window):
    new_window = (specific_window == np.max(specific_window))
    return new_window


def preprocess_mnist_data(x, y):
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float64") / 255
    y = y.reshape(-1, 1)
    y = np_utils.to_categorical(y)  # confusion
    new_y_shape = y.shape + (1,)
    y = np.reshape(y, new_y_shape)

    return x, y


def preprocess_cifar10_data(x, y):  # dimension (60k, 32, 32, 3)
    x = np.transpose(x, (0, 3, 1, 2))  # changed dimension (60k, 3, 32, 32)
    x = x.astype("float64") / 255
    y = np_utils.to_categorical(y)  # confusion
    new_y_shape = y.shape + (1, )
    y = np.reshape(y, new_y_shape)

    return x, y


def mnist_dataset_load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_mnist_data(x_train, y_train)
    x_test, y_test = preprocess_mnist_data(x_test, y_test)

    test_len = int(len(x_test) / 2)

    x_test_new = x_test[0:test_len]
    y_test_new = y_test[0:test_len]

    x_valid = x_test[test_len:]
    y_valid = y_test[test_len:]

    return x_train, y_train, x_valid, y_valid, x_test_new, y_test_new


def cifar10_dataset_load_and_preprocess():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # dimension (60k, 32, 32, 3)
    x_train, y_train = preprocess_cifar10_data(x_train, y_train)
    x_test, y_test = preprocess_cifar10_data(x_test, y_test)

    test_len = int(len(x_test) / 2)

    x_test_new = x_test[0:test_len]
    y_test_new = y_test[0:test_len]

    x_valid = x_test[test_len:]
    y_valid = y_test[test_len:]

    return x_train, y_train, x_valid, y_valid, x_test_new, y_test_new


def binary_cross_entropy_loss(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_loss_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k *
                                                       mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k *
                                                       mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches *
                                  mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches *
                                  mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def relu_derivative(x):
    x = np.where(x > 0, x, x * 0.01)
    return x
