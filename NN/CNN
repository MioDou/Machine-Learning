# -*- coding:utf-8 -*-

import numpy as np
from RNN.file.act import ReLUActivator,IdentityActivator

# 获取卷积区域
def get_patch(input_array,i,j,filter_width,filter_height,stride): #从输入数组中获取本次卷积的区域，自动适配输入为2D和3D的情况
    start_i = i * stride
    start_j = j * stride

    if input_array.ndim == 2:
        return input_array[ start_i : start_i + filter_height,
                            start_j : start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[ :,
                            start_i : start_i + filter_height,
                            start_j : start_j + filter_width]

# 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_i,max_j = i, j
    return max_i,max_j

#计算一个过滤器的卷积运算，输出一个二维数据。每个通道的输入是图片
def convolution(input_array, kernel_array, output_array, stride, bias):
    #channel_num = input_array.ndim
    output_width = output_array.shape[1]   # 获取输出的宽度
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]   # 过滤器的宽度
    kernel_height = kernel_array.shape [-2]   # 过滤器的宽度

    for i in range(output_height):
        for j in range(output_width):
            #获取输入的卷积区，每个通道里的卷积区矩阵与过滤器矩阵按对应元素相乘求和，将每个通道的和值再求和，结果加上偏量
            output_array[i][j] = (get_patch(input_array,i,j,kernel_width,kernel_height,stride) * kernel_array).sum() + bias

# 为数组增加Zero padding,补零圈数(指在原始图像周围补几圈0)，zp是步长
def padding(input_array,zp):
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:  # 如果输入有多个通道
            input_width = input_array.shape[2] # 获取输入的宽度
            input_height = input_array.shape[1] # 获取输入的高度
            input_depth = input_array.shape[0] # 获取输入的深度

            padded_array = np.zeros((input_depth,
                                     input_height + 2 * zp,
                                     input_width + 2 * zp)) # 定义一个补0后大小的全0矩阵

            padded_array[:,zp:zp + input_height,zp:zp + input_width] = input_array # 每个通道，将中间部分替换成输入，这样就变成了原矩阵周围补0的形式
            return padded_array

        elif input_array.ndim == 2:  # 如果输入只有1个通道
            input_width = input_array.shape[1] # 获取输入的宽度
            input_height = input_array.shape[0] # 获取输入的高度

            padded_array = np.zeros((input_height + 2 * zp,
                                     input_width + 2 * zp)) # 定义一个补0后大小的全0矩阵
            padded_array[zp:zp + input_height,zp:zp + input_width] = input_array  # 每个通道，将中间部分替换成输入，这样就变成了原矩阵周围补0的形式
            return padded_array


# 对numpy数组进行element wise操作（进行逐个元素的操作），op为函数
def element_wise_op(array,op):
    for i in np.nditer(array,op_flags = ['readwrite']):
        i[...] = op(i) # 将元素i传入op函数，返回值，再修改i

#过滤器，保留有卷积层的参数以及梯度，用梯度下降算法来更新参数。
class Filter(object):
    def __init__(self,width,height,depth):
        self.weights = np.random.uniform(-1e-4,1e-4,(depth,height,width))  # 随机初始化卷基层权重一个很小的值
        self.bias = 0  # 初始化偏量为0
        self.weights_gard = np.zeros(self.weights.shape)   #初始化权重梯度
        self.bias_gard = 0  # 初始化偏量梯度

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s'%(repr(self.weights),repr(self.bias))
    #读取权重
    def get_weights(self):
        return self.weights
    #读取偏量
    def get_bias(self):
        return self.bias
    #更新权重和偏量
    def update(self,learning_rate):
        self.weights -= learning_rate * self.weights_gard
        self.bias -= learning_rate * self.bias_gard

#卷积层
class ConvolutionLayer(object):
    #初始化一个卷积层，可以在构造函数中设置卷积层的超参数
    def __init__(self,input_width,input_height,channel_num,filter_width,filter_height,filter_num,zero_padding,stride,activator,learning_rate):
#        self.padded_input_array = padding(input_array,self.zero_padding)
#        self.input_array = input_array
#        self.delta_array = self.create_delta_array()
        self.input_width = input_width   #宽
        self.input_height = input_height   #高
        self.channel_num = channel_num      # 通道数=输入的深度=过滤器的深度
        self.filter_width = filter_width  # 过滤器的宽度
        self.filter_height = filter_height   # 过滤器的高度
        self.filter_num = filter_num   # 过滤器的数量
        self.zero_padding = zero_padding   #补0圈数
        self.stride = stride  # 步幅

        self.output_width = ConvolutionLayer.calculate_output_size(self.input_width,filter_width,zero_padding,stride)  #计算输出宽度
        self.output_height = ConvolutionLayer.calculate_output_size(self.input_height,filter_height,zero_padding,stride)   #计算输出高度
        self.output_array = np.zeros((self.filter_num,self.output_height,self.output_width))  # 创建输出三维数组。每个过滤器都产生一个二维数组的输出
        self.filters = []  #卷积层的过滤器

        for i in range(filter_num):
            self.filters.append(Filter(filter_width,filter_height,self.channel_num))

        self.activator = activator  #激活函数
        self.learning_rate = learning_rate #学习率

    def forward(self,input_array):
        """
            计算卷积层的输出
            输出结果保存在self.output_array
        """
        self.input_array = input_array  # 多个通道的图片，每个通道为一个二维图片
        self.padded_input_array = padding(input_array,self.zero_padding)   # 将输出用0补全
        for f in range(self.filter_num):  #每个过滤器都产生一个二维数组的输出
            filter_1 = self.filters[f]
            convolution(self.padded_input_array, filter_1.get_weights(), self.output_array[f], self.stride, filter_1.get_bias())
        # element_wise_op函数实现了对numpy数组进行按元素操作，并将返回值写回到数组中
        element_wise_op(self.output_array,self.activator.forward)

    def backward(self,input_array,sensitivity_array,activator):
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array,activator)
        self.bp_gradient(sensitivity_array)

    # 按照梯度下降，更新权重
    def update(self):
        for fil in self.filters:
            fil.update(self.learning_rate)

    def bp_sensitivity_map(self,sensitivity_array,activator):
        """
        将误差项传递到上一层。
        sensitivity_array: 本层的误差
        activator: 上一层的激活函数
         """
        expanded_array = self.expand_sensitivity_map(sensitivity_array) # 根据卷积步长，对原始sensitivity map进行补0扩展，扩展成如果步长为1的输出误差形状
        expanded_width = expanded_array.shape[2] #误差的宽度
        zp = int((self.input_width + self.filter_width - 1 - expanded_width) / 2) #计算步长
        padded_array = padding(expanded_array,zp) #补0

        self.delta_array = self.create_delta_array() # 初始化delta_array，保存传递到上一层的sensitivity_map

        for f in range(self.filter_num): # 遍历每一个过滤器。每个过滤器都产生多通道的误差，多个多通道的误差叠加
            filter_2 = self.filters[f]
            # 将滤波器每个通道的权重权重翻转180度。
            flipped_weights = np.array(list(map(lambda i:np.rot90(i,2), filter_2.get_weights())))

            delta_array = self.create_delta_array() # 计算与一个filter对应的delta_array

            for d in range(delta_array.shape[0]):  # 计算每个通道上的误差，存储在delta_array的对应通道上
                convolution(padded_array[f],flipped_weights[d],delta_array[d],1,0)
            self.delta_array += delta_array # 将每个滤波器每个通道产生的误差叠加
        # 将计算结果与激活函数的偏导数进行element-wise操作
        derivative_array = np.array(self.input_array) # 复制一个矩阵
        element_wise_op(derivative_array,activator.backward) # 逐个元素求偏导数。
        self.delta_array *= derivative_array # 误差乘以偏导数。得到上一层的误差

    # 对步长为S的sensitivity map相应的位置进行补0，将其还原成步长为1时的sensitivity map
    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0] # 获取误差项的深度

        # 确定扩展后sensitivity map的大小，即计算stride为1时sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)

        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))

        # 从原始sensitivity map拷贝误差值，每有拷贝的位置，就是要填充的0
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = i * self.stride
                expand_array[:, i_pos, j_pos] = sensitivity_array[:, i, j]
        return expand_array

    # 计算梯度。根据误差值，计算本层每个过滤器的w和b的梯度
    def bp_gradient(self,sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)

        for f in range(self.filter_num):     # 每个过滤器产生一个输出
            # 计算每个权重的梯度
            filter_3 = self.filters[f]
            for d in range(filter_3.weights.shape[0]):    # 过滤器的每个通道都要计算梯度
                convolution(self.padded_input_array[d],expanded_array[f],filter_3.weights_gard[d],1,0)
            filter_3.bias_grad = expanded_array[f].sum()   # 计算偏置项的梯度

    # 创建用来保存传递到上一层的sensitivity map的数组
    def create_delta_array(self):
        return np.zeros((self.channel_num,self.input_height,self.input_width))

    @staticmethod
    def calculate_output_size(input_size,filter_size,zero_padding,stride): # 确定卷积层输出的大小
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)

#最大值池化层(一个卷积区域取最大值，形成输出)
class MaxPoolingLayer(object):
    def __init__(self,input_weight,input_height,channel_num,filter_width,filter_height,stride):
#        self.delta_array = np.zeros(input_array.shape)
        self.input_width = input_weight
        self.input_height = input_height
        self.channel_num = channel_num
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride

        self.output_width = int((input_weight - filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        self.output_array = np.zeros((self.channel_num,self.output_height,self.output_width))

    # 前向计算
    def forward(self,input_array):
        for d in range(self.channel_num):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (get_patch(input_array[d],i,j,self.filter_width,self.filter_height,self.stride).max())
    # 反向传播，更新w,b
    def backward(self,input_array,sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_num):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(input_array[d],i,j,self.filter_width,self.filter_height,self.stride)
                    k,l = get_max_index(patch_array)
                    self.delta_array[d,i * self.stride + k,j * self.stride + l] = sensitivity_array[d,i,j]


def init_test():
    # 作为输入
    a = np.array([[[0, 1, 1, 0, 2],
                   [2, 2, 2, 2, 1],
                   [1, 0, 0, 2, 0],
                   [0, 1, 1, 0, 0],
                   [1, 2, 0, 0, 2]],
                  [[1, 0, 2, 2, 0],
                   [0, 0, 0, 2, 0],
                   [1, 2, 1, 2, 1],
                   [1, 0, 0, 0, 0],
                   [1, 2, 1, 1, 1]],
                  [[2, 1, 2, 0, 0],
                   [1, 0, 0, 1, 0],
                   [0, 2, 1, 0, 1],
                   [0, 1, 2, 2, 2],
                   [2, 1, 0, 0, 1]]])
    # 作为输出误差
    b = np.array([[[0, 1, 1],
                   [2, 2, 2],
                   [1, 0, 0]],
                  [[1, 0, 2],
                   [0, 0, 0],
                   [1, 2, 1]]])
    cl = ConvolutionLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    # 初始化第一层卷积层权重
    cl.filters[0].weights = np.array([[[-1, 1, 0],
                                       [0, 1, 0],
                                       [0, 1, 1]],
                                      [[-1, -1, 0],
                                       [0, 0, 0],
                                       [0, -1, 0]],
                                      [[0, 0, -1],
                                       [0, 1, 0],
                                       [1, -1, -1]]],dtype=np.float64)

    cl.filters[0].bias = 1 # 初始化第一层卷积层偏重

    # 初始化第二层卷积层权重
    cl.filters[1].weights = np.array([[[1, 1, -1],
                                       [-1, -1, 1],
                                       [0, -1, 1]],
                                      [[0, 1, 0],
                                       [-1, 0, -1],
                                       [-1, 1, 0]],
                                      [[-1, 0, 0],
                                       [-1, 0, 1],
                                       [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl

# 测试前向传播
def test():
    a,b,cl= init_test()
    cl.forward(a)  # 对输出进行以一次前向预测
    print(cl.output_array)

# 测试反向传播
def test_bp():
    a, b, cl = init_test()
    cl.backward(a,b,IdentityActivator())   # 对输出误差后向传播
    cl.update()  #更新权重
    print(cl.filters[0])
    print(cl.filters[1])

# 梯度检查
def gradient_check():
    error_function = lambda o: o.sum()  # 设计一个误差函数，取所有节点输出项之和
    a,b,cl = init_test()  # 计算forward值
    cl.forward(a)  # 对输入进行一次前向预测

    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape,dtype = np.float64)
    # 计算梯度
    cl.backward(a,sensitivity_array,IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d,i,j] = epsilon
                cl.forward(a)
                error_1 = error_function(cl.output_array)
                cl.filters[0].weights[d,i,j] -= 2 * epsilon
                cl.forward(a)
                error_2 = error_function(cl.output_array)
                expect_grad = int((error_1 - error_2) / (2 * epsilon))
                cl.filters[0].weights[d,i,j] += epsilon
                print('weights(%d,%d,%d):expected - actual %f - %f'%(d,i,j,expect_grad,cl.filters[0].weights_grad[d,i,j]))


def init_pool_test():
    a = np.array([[[1,1,2,4],
                   [5,6,7,8],
                   [3,2,1,0],
                   [1,2,3,4]],
                  [[0,1,2,3],
                   [4,5,6,7],
                   [8,9,0,1],
                   [3,4,5,6]]], dtype=np.float64)

    b = np.array(
        [[[1,2],
          [2,4]],
         [[3,5],
          [8,2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4,4,2,2,2,2)
    return a, b, mpl

def test_pool():
    a,b,mpl = init_pool_test()
    mpl.forward(a)
    print('input_array:\n%s\noutput_array:\n%s'% (a,mpl.output_array))

def test_pool_bp():
    a,b,mpl = init_pool_test()
    mpl.backward(a,b)
    print('input_array:\n%s\nsensitivity_array:\n%s\ndelta_array:\n%s'%(a,b,mpl.delta_array))
