# -*- coding:utf-8 -*-

"""create at 2019.8.5
    逻辑运算
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):

        self.W = []#定义权重
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers) - 2):# 权重初始化
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

            w = np.random.randn(layers[-2] + 1, layers[-1])
            self.W.append(w / np.sqrt(layers[-2]))

    def sigmoid (self, x):#sigmoid 激活函数
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_d(self, x):#sigmoid导数
        return x * (1 - x)


    def fit(self, X, y, epochs=1000, update=100):#训练过程  #整合X，y
        X = np.c_[X, np.ones((X.shape[0]))]

        for epochs in np.arange(0, epochs):
            for (x, target) in zip(X, y):# zip打包函数
                self.fit_partial(x, target)
            # 损失
            if epochs == 0 or (epochs + 1) % update == 0:
                loss = self.loss(X, y)
                print("[training] epochs={}, loss={:.3%}".format(epochs + 1, loss))

    def fit_partial(self, x, y):

        A = [np.atleast_2d(x)]#将x变为2维数组
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
        error = A[-1] - y
        D = [error * self.sigmoid_d(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_d(A[layer])
            D.append(delta)
        D = D[::-1]

        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss


input_nn = int(input("输入神经元数："))
input_nnn = int(input("输出神经元数："))
new_input_1 = list(map(int, input("数组0或1：").strip().split()))
new_input_2 = list(map(int, input("实际结果：").strip().split()))
#X = np.array([[0, 0],[0,1],[1, 0],[1, 1]])
#y = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])
X = np.array([new_input_1])
y = np.array([new_input_2])

#3层神经网络 ，神经元：输入2 隐藏4 输出2
nn = NeuralNetwork([input_nn, 10, input_nnn], alpha=0.1)

nn.fit(X, y, epochs=20000)#训练次数
# NOT c
print("\n")
print("[result]")
#x1 = X[0][0]
#x2 = X[0][1]

for (x, target) in zip(X, y):
    if len(new_input_1) > 1:
        i = len(new_input_1)-1
    else:
        i = 0
    predict_1 = nn.predict(x)[0][0]
    predict_2 = nn.predict(x)[0][i]
    #if abs(x1 - predict_x1) > 0.5:
      #  predict_1 = predict_x1
    if 1 - predict_1 < 0.5:
        error_1 = 1 - predict_1
    else:
        error_1 = predict_1
    if 1 - predict_2 < 0.5:
        error_2 = 1 - predict_2
    else:
        error_2 = predict_2
    print("data= {}, truth= [{} {}],  预测值_1 = {:.8f},最后数字预测值 = {:.8f},误差1={:.2%},最后数字误差={:.2%}".format(x,target[0],target[1], predict_1,predict_2,error_1,error_2))
