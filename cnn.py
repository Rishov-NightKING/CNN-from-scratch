import numpy as np

from utils import cross_entropy_loss


class ConvolutionalNeuralNetwork:
    def __init__(self):
        self.forward_layers = list()
        self.backward_layers = list()

    def add_layer(self, layer):
        self.forward_layers.append(layer)

    def train_model(self, x_train, y_train, learning_rate, number_of_epochs):
        self.backward_layers = self.forward_layers[::-1]
        number_of_train_samples = len(x_train)

        for epoch in range(number_of_epochs):
            per_epoch_loss = 0.0
            for j in range(number_of_train_samples):
                # forward propagation of a layer
                current_layer_predicted_output = x_train[j]
                for layer in self.forward_layers:
                    current_layer_predicted_output = layer.forward(current_layer_predicted_output)

                current_layer_predicted_output = np.where(current_layer_predicted_output > 0,
                                                          current_layer_predicted_output, 0.000000001)
                # print("forward passed successfully")
                # print("pred ", current_layer_predicted_output
                #print("real", real_value)
                # # error calculation
                # print(f"y_true_shape: {y_train[j].shape}  y_pred_shape: {current_layer_predicted_output.shape}")
                loss = cross_entropy_loss(y_true=y_train[j], y_pred=current_layer_predicted_output)
                #print(loss)
                per_epoch_loss += loss

                # backward propagation of a layer
                # softmax output

                # print("hi  ", current_layer_predicted_output.shape, "   ", real_value.shape)
                output_gradient = current_layer_predicted_output - y_train[j]  # y_prime - y
                # print("bye", output_gradient.shape)

                for layer in self.backward_layers:
                    output_gradient = layer.backward(output_gradient, learning_rate)

            # per_epoch_loss /= number_of_train_samples
            print(f'\nepoch: {epoch + 1} done....average loss: {per_epoch_loss}')

    def predict(self, x_test, y_test):
        number_of_test_samples = len(x_test)
        predicted_output = np.zeros((number_of_test_samples, 1))

        match_count = 0
        for i in range(number_of_test_samples):
            current_layer_predicted_output = x_test[i]
            for layer in self.forward_layers:
                current_layer_predicted_output = layer.forward(current_layer_predicted_output)

            predicted_output[i] = current_layer_predicted_output
            if predicted_output[i] == y_test[i]:
                match_count += 1

        print(f'Accuracy: {match_count / number_of_test_samples * 100}%')
        return current_layer_predicted_output
