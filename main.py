# from utils import file_read
# from convolutionalLayer import ConvolutionalLayer
# if __name__ == '__main__':
#     a = file_read('input.txt')
#     b = ConvolutionalLayer(input_shape=(28,28,1), filter_dimension=3, depth=5)

import numpy as np
from convolutionalLayer import ConvolutionalLayer
from maxPoolingLayer import MaxPoolingLayer
from fullyConnectedLayer import FullyConnectedLayer
from activationReLUlayer import ActivationReLuLayer
from softMaxLayer import SoftMaxLayer

if __name__ == "__main__":
    np.random.seed(1)
    input = np.random.randn(1, 28, 28)

    out1 = ConvolutionalLayer(input_shape=(1, 28, 28), number_of_output_channel=6, filter_dimension=5, stride=1,
                              padding=2).forward(input)
    out2 = ActivationReLuLayer().forward(out1)
    out3 = MaxPoolingLayer(input_shape=out2.shape, filter_dimension=2, stride=2).forward(out2)

    out4 = ConvolutionalLayer(input_shape=out3.shape, number_of_output_channel=12, filter_dimension=5, stride=1,
                              padding=0).forward(out3)
    out5 = ActivationReLuLayer().forward(out4)
    out6 = MaxPoolingLayer(input_shape=out5.shape, filter_dimension=2, stride=2).forward(out5)

    out7 = ConvolutionalLayer(input_shape=out6.shape, number_of_output_channel=100, filter_dimension=5, stride=1,
                              padding=0).forward(out6)
    out8 = ActivationReLuLayer().forward(out7)

    print(out8)

    out10 = FullyConnectedLayer(output_dimension=10).forward(out8)



    out11 = SoftMaxLayer().forward(out10)
    # print(out11)

    # layer2 = MaxPoolingLayer(input_shape=o1.shape, filter_dimension=2, stride=2)
    # o2 = layer2.forward(o1)

    # print(o2.shape)
    # print(o2)

    # # layer3 = FullyConnectedLayer(output_dimension=10)
    # # o3 = layer3.forward(o2)
    # # print(o3.shape)
    # # print(o3)
    #
    # layer4 = ActivationReLuLayer()
    # o4 = layer4.forward(input_data=o3)
    # # print(o4)
    #
    # layer5 = SoftMaxLayer()
    # o5 = layer5.forward(o4)
    # print(o5)
