# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# import sys
# sys.path.append('/Users/xsffsx/PycharmProjects/NeuralNetworksAndDeepLearning/neural-networks-and-deep-learning/src')

import network3
from network3 import FullyConnectedLayer, ConvPoolLayer, SoftmaxLayer
from network3 import Network

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print "starting..."
    training_data, validation_data, test_data = network3.load_data_shared()
    mini_batch_size = 10
    net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)
    ], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    print "complete."
