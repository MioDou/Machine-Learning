# -*- coding:utf-8 -*-

import numpy as np
from RNN.file.Convolution_NN import element_wise_op
from RNN.file.act import ReLUActivator,IdentityActivator
from functools import reduce

class RecurrentLayer:
    def __init__(self,input_width,state_width,activator,learning_rate):
        self.gradient_list = []
        self.delta_list = []
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((state_width,1)))
        self.U = np.random.uniform(-1e-4, 1e-4,(state_width, input_width))
        self.W = np.random.uniform(-1e-4, 1e-4,(state_width, state_width))


    def forward(self,input_array):
        self.times += 1
        state = (np.dot(self.U,input_array) + np.dot(self.W,self.state_list[-1]))
        element_wise_op(state,self.activator.forward)
        self.state_list.append(state)

    def backward(self,sensitivity_array,activator):
        self.calc_delta(sensitivity_array,activator)
        self.calc_gradient()

    def update(self):
        self.W -= self.learning_rate * self.gradient

    def calc_delta(self,sensitivity_array,activator):
        self.delta_list = []
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width,1)))
        self.delta_list.append(sensitivity_array)

        for k in range(self.times - 1,0,-1):
            self.calc_delta_k(k,activator)

    def calc_delta_k(self,k,activator):
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1],activator.backward)
        self.delta_list[k] = np.dot(np.dot(self.delta_list[k+1].T,self.W),np.diag(state[:,0])).T

    def calc_gradient(self):
        self.gradient_list = []
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros((self.state_width,self.state_width)))
        for t in range(self.times,0,-1):
            self.calc_gradient_t(t)
        self.gradient = reduce(lambda a,b:a + b,self.gradient_list,self.gradient_list[0])

    def calc_gradient_t(self,t):
        gradient = np.dot(self.delta_list[t],self.delta_list[t-1].T)
        self.gradient_list[t] = gradient

    def reset_state(self):
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((self.state_width,1)))

def data_set():
    x = [np.array([[1],[2],[3]]),
         np.array([[2],[3],[4]])]
    d = np.array([[1],[2]])
    return x,d

def gradient_check():
    error_function = lambda o:o.sum()

    rl = RecurrentLayer(3,2,IdentityActivator(),1e-3)

    x,d = data_set()
    rl.forward(x[0])
    rl.forward(x[1])
    sensitivity_array = np.ones(rl.state_list[-1].shape,dtype = np.float64)

    rl.backward(sensitivity_array,IdentityActivator())

    epslion = 10e-4
    for i in range(rl.W.shape[0]):
        for j in range(rl.W.shape[1]):
            rl.W[i,j] += epslion
            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            error_1 = error_function(rl.state_list[-1])
            rl.W[i,j] -= 2 * epslion

            rl.reset_state()
            rl.forward(x[0])
            rl.forward(x[1])
            error_2 = error_function(rl.state_list[-1])
            expect_grad = (error_1 - error_2) / (2 * epslion)
            rl.W[i,j] += epslion

            print('weights(%d,%d):expected - actual %f - %f' % (i,j,expect_grad,rl.gradient[i,j]))

def test():
    l = RecurrentLayer(3, 2, ReLUActivator(), 1e-3)
    x, d = data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d, ReLUActivator())
    return l
