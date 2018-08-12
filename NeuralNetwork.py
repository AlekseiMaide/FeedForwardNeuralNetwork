# -*- coding: UTF-8 -*-

import numpy as np


def sigmoid_function(x, derivative=False):
    """
    Sigmoid function
    “x” is the input and “y” the output, the nonlinear properties of this function means that
    the rate of change is slower at the extremes and faster in the centre. Put plainly,
    we want the neuron to “make its mind up” instead of indecisively staying in the middle.
    :param x: Float
    :param derivative: Boolean
    :return: Float
    """
    if (derivative):
        return x * (1 - x)  # Derivative using the chain rule.
    else:
        return 1 / (1 + np.exp(-x))


# create dataset for XOR problem
input_data = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
ideal_output = np.array([[0.0], [1.0], [1.0], [0.0]])

# initialize variables
learning_rate = 0.1
epoch = 50000  # number or iterations basically - One round of forward and back propagation is called an epoch

# get the second element from the numpy array shape field to detect the count of features for input layer
input_layer_neurons = input_data.shape[1]
hidden_layer_neurons = 3  # number of hidden layer neurons
output_layer_neurons = 1  # number of output layer neurons

# init weight & bias
weights_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(1, hidden_layer_neurons)
weights_output = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
bias_output = np.random.uniform(1, output_layer_neurons)

for i in range(epoch):

    # forward propagation
    hidden_layer_input_temp = np.dot(input_data,
                                     weights_hidden)  # matrix dot product to adjust for weights in the layer
    hidden_layer_input = hidden_layer_input_temp + bias_hidden  # adjust for bias
    hidden_layer_activations = sigmoid_function(hidden_layer_input)  # use the activation function
    output_layer_input_temp = np.dot(hidden_layer_activations, weights_output)
    output_layer_input = output_layer_input_temp + bias_output
    output = sigmoid_function(output_layer_input)  # final output

    # backpropagation (where adjusting of the weights happens)
    error = ideal_output - output  # error gradient
    if (i % 1000 == 0):
        print("Error: {}".format(np.mean(abs(error))))

    # use derivatives to compute slope of output and hidden layers
    slope_output_layer = sigmoid_function(output, derivative=True)
    slope_hidden_layer = sigmoid_function(hidden_layer_activations, derivative=True)

    # calculate deltas
    delta_output = error * slope_output_layer
    error_hidden_layer = delta_output.dot(weights_output.T)  # calculates the error at hidden layer
    delta_hidden = error_hidden_layer * slope_hidden_layer

    # change the weights
    weights_output += hidden_layer_activations.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_hidden += input_data.T.dot(delta_hidden) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

print(output)

# np.save('XOR_MODEL', output)
print("------------")
print(weights_hidden)
print("------------")
print(weights_output)

if __name__ == '__main__':
    loaded_model = np.load('XOR_MODEL.npy')
    # print(loaded_model)

    print(loaded_model == output)  # equality is true
    print(loaded_model == output)  # not the same instance/memory location
