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
    def __init__(self, layers, activation='TanH'):

        if activation == 'Sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'TanH':
            self.activation = TanH
            self.activation_derivative = TanH_derivative

        self.weights = []
        # 初始化 权值范围 [-0.25,0.25)
        for i in range(1, len(layers)-1):  #随机 生成weight 和  bias
            # [0,1) * 2 - 1 => [-1,1) => * 0.25 => [-0.25,0.25)
            self.weights.append( (2*np.random.random((layers[i-1] + 1, layers[i] + 1 ))-1 ) * 0.25 )
            self.weights.append( (2*np.random.random((layers[i] + 1, layers[i+1] ))-1 ) * 0.25 )
        # for i in range(0, len(layers)-1):
        #     m = layers[i]  # 第i层节点数
        #     n = layers[i+1]  # 第i+1层节点数
        #     wm = m + 1
        #     wn = n + 1
        #     if i == len(layers)-2:
        #         wn = n
        #     weight = np.random.random((wm, wn)) * 2 - 1
        #     self.weights.append(0.25 * weight)

#参数X为样本数据，y为标签数据，learning_rate 为学习率默认0.2，epochs 为迭代次数默认值10000
    def fit(self, X, y, learning_rate=0.2, epochs = 10000):
        X = np.atleast_2d(X) #转成2维
        # temp = np.ones([X.shape[0], X.shape[1]+1])
        # temp[:,0:-1] = X   # 最后一列的1是常数
        # X = temp
        X = np.column_stack((X, np.ones(len(X))))
        y = np.array(y)

        for k in range(epochs):  # 随机选择下标
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            # 正向计算
            for l in range(len(self.weights)): # 进过两次运算后计算出输出层
                a.append(self.activation( np.dot(a[l], self.weights[l])) )

            # 反向传播
            error = y[i] - a[-1]
            deltas = [error * self.activation_derivative(a[-1])] # 输出层误差
            # starting Back_propagation
            layerNum = len(a) - 2

            for j in range(layerNum, 0, -1): # 倒数第二层开始
                deltas.append(deltas[-1].dot(self.weights[j].T) * self.activation_derivative(a[j])) #隐藏层误差
                # deltas.append(deltas[-(layerNum+1-j)].dot(self.weights[j].T) * self.activation_deriv(a[j]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i]) 
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)   #更新weights

    def predict(self, input_sum):
        input_sum = np.array(input_sum)
        temp = np.ones(input_sum.shape[0] + 1)
        temp[0:-1] = input_sum
        a = temp
        for l in range(0, len(self.weights)):  #不用保留之前层的数值,只保留最后一层数值,并且放入到sigmoid函数中
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == "__main__":
    nn = NeuralNetwork([2, 2, 1], 'TanH')
    # nn = DeepNeuralNetwork([2, 2, 1], 'TanH')
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(x, y)
    for i in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(i, nn.predict(i))
