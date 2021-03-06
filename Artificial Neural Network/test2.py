# -*- coding:utf-8 -*-
"""
  改变数据集的情况
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Test_3.planner_utils import plot_decision_boundary,load_planar_dataset,load_extra_datasets
from Test_3.testCase import predict_test_case,layer_sizes_test_case,initialize_parameters_test_case,nn_model_test_case,forward_propagation_test_case,backward_propagation_test_case,compute_cost_test_case,update_parameters_test_case


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

noisy_circles, noisy_moons, blobs, gaussian_quartiles, no_structure = load_extra_datasets()

data = {"noisy_circles": noisy_circles,
        "noisy_moons": noisy_moons,
        "blobs": blobs,
        "gaussian_quartiles": gaussian_quartiles}

data_set = "noisy_moons"

X, Y = data[data_set]
X, Y = X.T, Y.reshape(1, Y.shape[0])

if data_set == "blobs":
    Y = Y % 2

plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
plt.show()

def layers_sizes(x,y):
    n_x = x.shape[0]
    n_h = 4
    n_y = y.shape[0]
    return n_x,n_h,n_y

def initializer_parameters(n_x, n_H, n_y):
    np.random.seed(2)
    W1= np.random.randn(n_H, n_x) * 0.01
    b1=np.zeros((n_H, 1))
    W2= np.random.randn(n_y, n_H) * 0.01
    b2=np.zeros((n_y,1))

    assert (W1.shape == (n_H, n_x))
    assert (b1.shape == (n_H, 1))
    assert (W2.shape == (n_y, n_H))
    assert (b2.shape == (n_y, 1))

    parameter={'W1':W1,
                'b1':b1,
                'W2':W2,
                'b2':b2}
    return parameter

def forward_propagation(x, parameter):
    W1=parameter["W1"]
    b1=parameter["b1"]
    W2=parameter["W2"]
    b2=parameter["b2"]

    Z1 = np.dot(W1, x) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, x.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache

def compute_cost(A2, y):

    M = y.shape[1]
    cost= -1 / M * np.sum(np.multiply(y, np.log(A2)) + np.multiply((1 - y), np.log(1 - A2)))

    cost=np.squeeze(cost)           # makes sure cost is the dimension we expect.

    assert(isinstance(cost,float))
    return cost

def backward_propagation(parameter, cache, x, y):
    M = x.shape[1]

    W1=parameter['W1']
    W2=parameter['W2']

    A1=cache['A1']
    A2=cache['A2']

    dZ2= A2 - y              #n_y,m
    dW2= np.dot(dZ2,A1.T) / M               #n_y,n_h
    db2= np.sum(dZ2,axis=1,keepdims=True) / M              #n_y,1
    dZ1=np.multiply(np.dot(W2.T,dZ2),1 - np.power(A1, 2))                     #n_h,m
    dW1= np.dot(dZ1, x.T) / M                               #n_h,n_x
    db1= np.sum(dZ1,axis=1,keepdims=True) / M              #n_h,1

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameter, grads, learning_rate=1.2):
    W1 = parameter["W1"]
    b1 = parameter["b1"]
    W2 = parameter["W2"]
    b2 = parameter["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    ###梯度更新，每迭代一次更新一次###
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameter = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameter

def nn_model(x, y, n_H, num_iterations=10000, print_cost=False):

    np.random.seed(3)
    n_x = layers_sizes(x, y)[0]
    n_y = layers_sizes(x, y)[2]

    parameter =initializer_parameters(n_x, n_H, n_y)
 #   W1 = parameter["W1"]
 #   b1 = parameter["b1"]
 #   W2 = parameter["W2"]
 #   b2 = parameter["b2"]

    for k in range(0, num_iterations):
        A2,cache=forward_propagation(x, parameter)#前向传播节点
        cost = compute_cost(A2, y)#计算损失函数
        grads=backward_propagation(parameter, cache, x, y)#计算后向传播梯度
        parameter=update_parameters(parameter, grads, learning_rate=1.2)#使用梯度更新W，b一次

        if print_cost:
            if k % 1000 == 0:
                print ("Cost after iteration %i: %f" % (k, cost))
    return parameter

def predict(parameter, x):

    A2, cache = forward_propagation(x, parameter)
    prediction = (A2 > 0.5)
    return prediction


parameters = nn_model(X, Y, n_H = 4, num_iterations=10000, print_cost=True)

#绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.show()
