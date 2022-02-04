import numpy as np

from utils import binary_cross_entropy_loss, binary_cross_entropy_loss_derivative


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.layers = list()

    def add_layer(self, layer):
        self.layers.append(layer)

    def train_model(self, x_train, y_train, learning_rate, number_of_epochs):
        number_of_train_samples = len(x_train)

        for epoch in range(number_of_epochs):
            per_epoch_loss = 0.0
            for j in range(number_of_train_samples):
                # forward propagation of a layer
                current_layer_predicted_output = x_train[j]
                for layer in self.layers:
                    current_layer_predicted_output = layer.forward(current_layer_predicted_output)

                # error calculation
                print(f"y_true_shape: {y_train[j].shape}  y_pred_shape: {current_layer_predicted_output.shape}")
                per_epoch_loss += binary_cross_entropy_loss(y_true=y_train[j], y_pred=current_layer_predicted_output)

                # backward propagation of a layer
                output_gradient = binary_cross_entropy_loss_derivative(y_true=y_train[j],
                                                                       y_pred=current_layer_predicted_output)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward(output_gradient, learning_rate)

            per_epoch_loss /= number_of_train_samples
            print(f'epoch: {epoch + 1} done....average loss: {per_epoch_loss}')

    def predict(self, x_test, y_test):
        number_of_test_samples = len(x_test)
        predicted_output = np.zeros((number_of_test_samples, 1))

        match_count = 0
        for i in range(number_of_test_samples):
            current_layer_predicted_output = x_test[i]
            for layer in self.layers:
                current_layer_predicted_output = layer.forward(current_layer_predicted_output)

            predicted_output[i] = current_layer_predicted_output
            if predicted_output[i] == y_test[i]:
                match_count += 1

        print(f'Accuracy: {match_count/number_of_test_samples * 100}%')
        return current_layer_predicted_output
