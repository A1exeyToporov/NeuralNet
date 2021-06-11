
from numpy import random, divide, zeros, dot, sum, squeeze, sqrt
import NeuralNet.Activations as act
import NeuralNet.Activations_derivatives as act_d
from NeuralNet import Costs

class NeuralNetwork:

    def __init__(self, input_size):
        self.shapes = [input_size]
        self.dims = []
        self.input_size = input_size
        self.num_layers = 0
        self.parameters = {}
        self.parameters_list = []
        self.activation_functions = {}
        self.caches = ({}, {})
        self.gradients = {}
        self.loss = None

    def __init_weights(self):
        for i in range(len(self.shapes)-1):
            self.parameters['W' + str(i+1)] = random.randn(self.shapes[i+1], self.shapes[i]) / sqrt(self.shapes[i])
            self.parameters['b' + str(i+1)] = zeros((self.shapes[i+1], 1))
        return self.parameters

    def get_layers_shape(self):
        for i in self.parameters.keys():
            print(i, self.parameters[i].shape)

    def get_num_layers(self):
        return self.num_layers

    def get_parameters(self):
        return self.parameters

    def add_layer(self, layer_size, activation='relu'):
        self.num_layers += 1
        self.shapes.append(layer_size)
        self.dims.append(layer_size)
        self.activation_functions['L' + str(self.num_layers)] = activation

    def __forward_pass(self, x):
        A = x.T
        self.caches[1]['A' + str(0)] = A
        for i in range(1, self.num_layers + 1):
            W = self.parameters['W' + str(i)]
            b = self.parameters['b' + str(i)]
            Z = dot(W, A) + b
            self.caches[0]['Z' + str(i)] = Z
            if self.activation_functions['L' + str(i)] == 'relu':
                A = act.relu(Z)
            if self.activation_functions['L' + str(i)] == 'sigmoid':
                A = act.sigmoid(Z)
            else:
                pass
            self.caches[1]['A' + str(i)] = A
        return A

    def __backward_pass(self, Y, A):
        linear_cache, activation_cache = self.caches
        Y = Y.reshape(A.shape)
        m = A.shape[1]
        if self.loss == 'binary':
            Yh = - (divide(Y, A) - divide(1 - Y, 1 - A))
        if self.loss == 'MSE':
            m = Y.shape[1]
            Yh = (A - Y)*(1/m)
        if self.activation_functions['L' + str(self.num_layers)] == 'sigmoid':
            dZ = act_d.sigmoid_derivative(Yh, linear_cache['Z' + str(self.num_layers)])
        if self.activation_functions['L' + str(self.num_layers)] == 'relu':
            dZ = act_d.relu_derivative(Yh, linear_cache['Z' + str(self.num_layers)])

        dW = dot(dZ, activation_cache['A' + str(self.num_layers - 1)].T) / m
        db = sum(dZ, axis=1, keepdims=True) / m
        dA_prev = dot(self.parameters['W' + str(self.num_layers)].T, dZ)

        self.gradients["dW" + str(self.num_layers)] = dW
        self.gradients["db" + str(self.num_layers)] = db

        for i in reversed(range(1, self.num_layers)):
            if self.activation_functions['L' + str(i)] == 'sigmoid':
                dZ = act_d.sigmoid_derivative(dA_prev, linear_cache['Z'+str(i)])
            if self.activation_functions['L' + str(i)] == 'relu':
                dZ = act_d.relu_derivative(dA_prev, linear_cache['Z'+str(i)])

            dW = dot(dZ, activation_cache['A'+str(i-1)].T) / m
            db = sum(dZ, axis=1, keepdims=True) / m
            dA_prev = dot(self.parameters['W'+str(i)].T, dZ)

            self.gradients["dW" + str(i)] = dW
            self.gradients["db" + str(i)] = db
        return self.gradients

    def __update_parameters(self, learning_rate):
        for i in range(self.num_layers):
            self.parameters["W" + str(i + 1)] = self.parameters["W" + str(i + 1)] - learning_rate * self.gradients["dW" + str(i + 1)]
            self.parameters["b" + str(i + 1)] = self.parameters["b" + str(i + 1)] - learning_rate * self.gradients["db" + str(i + 1)]
        return self.parameters

    def fit(self, X, Y, loss, num_iterations=100, learning_rate=0.1, verbose = 10):

        self.loss = loss
        self.parameters = self.__init_weights()

        for i in range(num_iterations):
            A = self.__forward_pass(X)
            if self.loss == 'MSE':
                cost = Costs.MSE_cost(A, Y)
            if self.loss == 'binary':
                cost = Costs.binary_cost(A, Y)
            self.gradients = self.__backward_pass(Y, A)
            self.parameters = self.__update_parameters(learning_rate)
            if i % verbose == 0 or i == num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, squeeze(cost)))

    def predict(self, X):
        if self.loss == 'binary':
            A = self.__forward_pass(X)
            for i in range(0, A.shape[1]):
                if A[0, i] > 0.5:
                    A[0, i] = 1
                else:
                    A[0, i] = 0

        if self.loss == 'MSE':
            A = self.__forward_pass(X)

        return A

    def accuracy(self, Y_true, Y_pred):
        Y_true = Y_true.reshape(Y_pred.shape)
        m = Y_true.shape[1]
        return sum((Y_true == Y_pred)/m)

