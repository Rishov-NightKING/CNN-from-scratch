import numpy as np

from utils import cross_entropy_loss


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.forward_layers = list()
        self.backward_layers = list()

    def add_layer(self, layer):
        self.forward_layers.append(layer)

    def train_model(self, x_train, y_train, x_valid, y_valid, learning_rate, number_of_epochs):
        self.backward_layers = self.forward_layers[::-1]
        number_of_train_samples = len(x_train)

        for epoch in range(number_of_epochs):
            for j in range(number_of_train_samples):
                # forward propagation of a layer
                current_layer_predicted_output = x_train[j]
                for layer in self.forward_layers:
                    current_layer_predicted_output = layer.forward(current_layer_predicted_output)

                # backward propagation of a layer
                # softmax output
                output_gradient = current_layer_predicted_output - y_train[j]  # y_prime - y

                for layer in self.backward_layers:
                    output_gradient = layer.backward(output_gradient, learning_rate)

            # validation
            per_epoch_accuracy, per_epoch_avg_loss = self.predict(x_valid, y_valid)
            print(
                f'\n    Validation for epoch-{epoch + 1}:\nAccuracy: {per_epoch_accuracy}%     '
                f'Average loss: {per_epoch_avg_loss}\n')

    def predict(self, x_test, y_test):
        number_of_test_samples = len(x_test)

        match_count = 0
        prediction_loss = 0
        for i in range(number_of_test_samples):
            current_layer_predicted_output = x_test[i]
            for layer in self.forward_layers:
                current_layer_predicted_output = layer.forward(current_layer_predicted_output)

            predicted_label = np.argmax(current_layer_predicted_output)
            actual_label = np.argmax(y_test[i])

            if predicted_label == actual_label:
                match_count += 1

            prediction_loss += cross_entropy_loss(y_true=y_test[i], y_pred=current_layer_predicted_output)

        prediction_loss /= number_of_test_samples
        accuracy = match_count / number_of_test_samples * 100\

        return accuracy, prediction_loss
