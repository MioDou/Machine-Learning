# -*- coding:utf-8 -*-

import numpy as np

#ReLU激活函数，激活器
class ReLUActivator(object):
    def forward(self,weight_input): #前向传播，计算输出
        return max(0,weight_input)

    def backward(self,output): #反向传播，计算导数
        return 1 if output > 0 else 0

# IdentityActivator激活器.f(x)=x
class IdentityActivator(object):
    def forward(self,weight_input):
        return weight_input

    def backward(self,output):
        return 1

#sigmoid激活函数
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

#TanH激活函数
class TanHActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output
