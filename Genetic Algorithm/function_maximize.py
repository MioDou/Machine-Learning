# -*- coding:utf-8 -*-
import numpy as np
import math
from scipy.optimize import fmin,fminbound
import matplotlib.pyplot as plt

def obj_function(x1,x2,X):  #变量均可改
    Y = 4 * x1 * x1 * x1 + 2 * x2 * x2 + 3 * X
    return Y

def b2d(bin): #二进制->十进制
    res = 0
    max_index = len(bin) - 1
    for i in range(len(bin)):
        if bin[i] == 1:
            res = res + 2**(max_index-i)
    return res

def ecd_length(low_bound, up_bound, precision):#  染色体长度
    divide = (up_bound - low_bound) / precision
    for i in range(10000):
        if 2**i < divide < 2**(i + 1):
            return i+1
    return -1

def dcd_chromosome(low_bound, up_bound, length, chromosome):  #解码二进制的染色体
    return low_bound + b2d(chromosome) * (up_bound - low_bound) / (2 ** length - 1)

def inti_pop(length, population_size):  #  定义初始染色体
    chromosomes = np.zeros((population_size,length),dtype=np.int8)
    for i in range(population_size):
        chromosomes[i] = np.random.randint(0,2,length)              #随机数[0,2)之间的整数
    return chromosomes

#参数
pop_size = 500 #population种群大小
is_elitist = True  #is elitist 精英
generations =  1000 # 种群进化多少代
precisions = 0.01  #精确值

#x1 区间
x1_low_bound = -2
x1_up_bound = 4
# x2 区间
x2_low_bound = -1
x2_up_bound = 3
# X 区间
X_low_bound = -2
X_up_bound = 2
# x1,x2,X对应的染色体长度和染色体初始化
chromosome_length_1 = ecd_length(x1_low_bound, x1_up_bound, precisions)
chromosome_length_2 = ecd_length(x2_low_bound, x2_up_bound, precisions)
chromosome_length_3 = ecd_length(X_low_bound, X_up_bound, precisions)
populations_1 = inti_pop(chromosome_length_1, pop_size)
populations_2 = inti_pop(chromosome_length_2, pop_size)
populations_3 = inti_pop(chromosome_length_3, pop_size)

Max_fit = 0  #最大值
#Min_fit = 0  #最小值
Best_Generation_max = 0  #获得最大值时的进化的代数
Best_chromosome = [0 for x in range(pop_size)]
cross_rate  = 0.6  # 交叉概率
mutate_rate = 0.01  # 变异概率

def fitness(population_1, population_2,population_3):#   适应度值
    Fitness_val = [0 for x in range(pop_size)]
    for i in range(pop_size):
        Fitness_val[i]= obj_function(dcd_chromosome(x1_low_bound, x1_up_bound, chromosome_length_1, population_1[i]),
                                     dcd_chromosome(x2_low_bound, x2_up_bound, chromosome_length_2, population_2[i]),
                                     dcd_chromosome(X_low_bound, X_up_bound, chromosome_length_3, population_3[i]))
    return Fitness_val

def rank(fitness_val,populations,cur_generation,chromosome_length):
    global Max_fit,Best_Generation_max,Best_chromosome
    fitness_sum = [0 for i in range(len(populations))]
    for i in range(len(populations)):       #排序
        min_index = 1
        for j in range(i+1, pop_size):
            if fitness_val[j] < fitness_val[min_index]:
                min_index = j
        if min_index != i:
            tmp = fitness_val[i]
            fitness_val[i] = fitness_val[min_index]
            fitness_val[min_index] = tmp

            tmp_list = np.zeros(chromosome_length)
            for k in range(chromosome_length):
                tmp_list[k] = populations[i][k]
                populations[i][k] = populations[min_index][k]
                populations[min_index][k] = tmp_list[k]
    for l in range(len(populations)):
        if l == 1:
            fitness_sum[l] = fitness_val[l]
        else:
            fitness_sum[l] = fitness_val[l] + fitness_val[l-1]

    if fitness_val[-1] > Max_fit:
        Max_fit = fitness_val[-1]
        Best_Generation_max = cur_generation
        for m in range(chromosome_length):
            Best_chromosome[m] = populations[-1][m]
    return fitness_sum

def select(populations, fitness_sum, chromosome_length, is_Elitist):   # 根据当前种群，选择新一代染色体
    population_new = np.zeros((pop_size, chromosome_length), dtype=np.int8)
    for i in range(pop_size):
        rnd = np.random.rand() * fitness_sum[-1]
        first = 0
        last = pop_size - 1
        mid = (first + last) // 2
        idx = -1
        while first <= last:
            if rnd > fitness_sum[mid]:
                first = mid
            elif rnd < fitness_sum[mid]:
                last = mid
            else:
                idx = mid
                break
            if last - first == 1:
                idx = last
                break
            mid = (first + last) // 2

        for j in range(chromosome_length):
            population_new[i][j] = populations[idx][j]

    if is_Elitist :  # 是否精英，强制保留适应度函数值最高的染色体
        p = pop_size - 1
    else:
        p = pop_size
    for l in range(p):
        for m in range(chromosome_length):
            populations[l][m] = populations[l][m]

def crossover(populations,chromosome_length):
    for i in range(0, pop_size, 2):
        rnd = np.random.rand()
        if rnd < cross_rate:
            rnd1 = int(math.floor(np.random.rand() * chromosome_length))
            rnd2 = int(math.floor(np.random.rand() * chromosome_length))
        else:
            continue
        if rnd1 <= rnd2:
           cross_position1 = rnd1   #交叉
           cross_position2 = rnd2
        else:
           cross_position1 = rnd2
           cross_position2 = rnd1
        for j in range(cross_position1,cross_position2):
            tmp = populations[i][j]
            populations[i][j] = populations[i+1][j]
            populations[i+1][j] = tmp

def mutation(populations,chromosome_length):
    for i in range(pop_size):
        rnd = np.random.rand()
        if rnd < mutate_rate:
            mutate_position = int(math.floor(np.random.rand() * chromosome_length))
        else:
            continue
        populations[i][mutate_position] = 1 - populations[i][mutate_position]

for i in range(generations):
    print("Generation {} ".format(i))
    fitness_value= fitness(populations_1, populations_2,populations_3)

    fitness_sum_1 = rank(fitness_value, populations_1, i, chromosome_length_1)
    fitness_sum_2 = rank(fitness_value, populations_2, i, chromosome_length_2)
    fitness_sum_3 = rank(fitness_value, populations_3, i, chromosome_length_3)

    select(populations_1, fitness_sum_1, chromosome_length_1,is_elitist)
    select(populations_2, fitness_sum_2, chromosome_length_2,is_elitist)
    select(populations_3, fitness_sum_3, chromosome_length_3,is_elitist)

    crossover(populations_1,chromosome_length_1)
    crossover(populations_2,chromosome_length_2)
    crossover(populations_3, chromosome_length_3)

    mutation(populations_1,chromosome_length_1)
    mutation(populations_2,chromosome_length_2)
    mutation(populations_3, chromosome_length_3)

    print("Best_Generation", Best_Generation_max)
    print("Max_fit", Max_fit)
print(" ")
print("=======")
print("获得最大值的一代：", Best_Generation_max)
print("最大值：%.4f" % Max_fit)
print("=======")
