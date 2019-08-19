# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import math
def load_dataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],          # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]                              #类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList, classVec

def create_list(dataset):
    cre_set = set([])  #创建一个不重复的空列表
    for doc in dataset:
        cre_set = cre_set | set(doc)  #取并集
    return list(cre_set)

def create_vec(List, input_set):
    vec = [0] * len(List)
    for word in input_set:
        if word in List:
            vec[List.index(word)] = 1
        else:
            print("the word :%s in not ---"%word)
    return vec

def train(train_vec,category):
    doc_num = len(train_vec)
    words_num = len(train_vec[0])

    pr_A = sum(category) / float(doc_num)
    p0 = np.ones(words_num)
    p1 = np.ones(words_num)
    p0_Denominator = 0.0;p1_Denominator = 0.0
    for i in range(doc_num):
        if category[i] == 1:
            p1 += train_vec[i]
            p1_Denominator += sum(train_vec[i])
        else:
            p0 += train_vec[i]
            p0_Denominator += sum(train_vec[i])

    pr_1 = np.log(p1 / p1_Denominator)
    pr_0 = np.log(p0 / p0_Denominator)
    return pr_0,pr_1,pr_A

def classifier(data_vec, p0vec, p1vec, class1):
    p1 = reduce(lambda x, y: x * y, data_vec * p1vec) * class1  # 对应元素相乘
    p0 = reduce(lambda x, y: x * y, data_vec * p0vec) * (1.0 - class1)
    print("p1:",p1)
    print("p0:",p0)
    if [p1] > [p0]:
        return 1
    else:
        return 0

def test():
    data,classV = load_dataset()
    vec_list = create_list(data)
    trainMat = []
    for ins in data:
        trainMat.append(create_vec(vec_list,ins))
    pr0,pr1,prA = train(np.array(trainMat),np.array(classV))
    #testEntry = ['love','to','food']
    testEntry = ['stupid','garbage']
    doc = np.array(create_vec(vec_list,testEntry))
    if classifier(doc,pr0,pr1,prA) :
        print(testEntry,'so rude')
    else:
        print(testEntry,'not rude')


if __name__ == '__main__':
    test()
