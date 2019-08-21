# -*- coding:utf-8 -*-

import numpy as np

def TanH(input_sum):
    return np.tanh(input_sum)
def TanH_derivative(input_sum):
    return 1.0 - np.tanh(input_sum) * np.tanh(input_sum)
def sigmoid(input_sum):
    return 1/(1+np.exp(-input_sum))
def sigmoid_derivative(input_sum):
    return input_sum*(1-input_sum)

class NeuralNetwork:
    def __init__(self, layers, activation='Sigmoid'):

        if activation == 'Sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'TanH':
            self.activation = TanH
            self.activation_derivative = TanH_derivative

        self.weights = []
        for I in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[I - 1] + 1, layers[I] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[I] + 1, layers[I + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs = 10000):
        X = np.atleast_2d(X)
        X = np.column_stack((X, np.ones(len(X))))
        y = np.array(y)

        for k in range(epochs):
            I = np.random.randint(X.shape[0])
            a = [X[I]]
            for l in range(len(self.weights)):
                a.append(self.activation( np.dot(a[l], self.weights[l])) )

            error = y[I] - a[-1]
            deltas = [error * self.activation_derivative(a[-1])]
            layerNum = len(a) - 2

            for j in range(layerNum, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_derivative(a[j]))
            deltas.reverse()
            for I in range(len(self.weights)):
                layer = np.atleast_2d(a[I])
                delta = np.atleast_2d(deltas[I])
                self.weights[I] += learning_rate * layer.T.dot(delta)

    def predict(self, input_sum):
        input_sum = np.array(input_sum)
        temp = np.ones(input_sum.shape[0] + 1)
        temp[0:-1] = input_sum
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == "__main__":
    nn = NeuralNetwork([2, 2, 1], 'TanH')
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    nn.fit(x, Y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(i, nn.predict(i))
