# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# import sys
# sys.path.append('/Users/xsffsx/PycharmProjects/NeuralNetworksAndDeepLearning/neural-networks-and-deep-learning/src')
from pydoc import help

import mnist_loader
import network2

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    net.large_weight_initializer()
    print "starting..."
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=test_data)
    print "complete."