# -*- coding:utf-8 -*-

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

#定义函数1
def function1(x):
    y1 = -x**2
    return y1

#定义函数2
def function2(x):
    y2 = -(x - 2) ** 2
    return y2

#Function to find index of list
# #查找列表指定元素的索引
def index_of(a, b):
    for i in range(0, len(b)):
        if b[i] == a:
            return i
    return -1
#Function to sort by values
#  函数根据指定的值列表排序
""" list1=[1,2,3,4,5,6,7,8,9]   
    value=[1,5,6,7]   
    sort_list=[1,5,6,7]
"""


def sort_by_values(list1, values):
    sorted_list = []
    while len(sorted_list)!=len(list1):        # 当结果长度不等于初始长度时，继续循环
        if index_of(min(values),values) in list1:            # 标定值中最小值在目标列表中时
            sorted_list.append(index_of(min(values),values))        #     将标定值的最小值的索引追加到结果列表后面
        values[index_of(min(values),values)] = math.inf    #      将标定值的最小值置为无穷小,即删除原来的最小值,移向下一个

    return sorted_list


def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]    # 种群中所有个体的sp进行初始化 这里的len(value1)=pop_size
    fronts = [[]]    # 分层集合,二维列表中包含第n个层中,有那些个体
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    # 评级
    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:                    # 同时如果q不属于sp将其添加到sp中
                    S[p].append(q)            # 如果q支配p
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):                # 则将np+1
                    n[p] = n[p] + 1
        if n[p]==0:            # 找出种群中np=0的个体
            rank[p] = 0            # 将其从pt中移去
            if p not in fronts[0]:                # 如果p不在第0层中                # 将其追加到第0层中
                fronts[0].append(p)
    i = 0
    while fronts[i]:        # 如果分层集合为不为空，
        Q=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        fronts.append(Q)
    del fronts[len(fronts) - 1]
    return fronts

def crowding_distance(values1, values2, fronts):
    distance = [0 for i in range(0, len(fronts))]    # 初始化个体间的拥挤距离
    sorted1 = sort_by_values(fronts, values1[:])
    sorted2 = sort_by_values(fronts, values2[:])    # 基于目标函数1和目标函数2对已经划分好层级的种群排序
    distance[0] = 4444444444444444
    distance[len(fronts) - 1] = 4444444444444444
    for k in range(1, len(fronts) - 1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1, len(fronts) - 1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance     #     返回拥挤距离 #函数进行交叉

def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2) #函数进行变异操作

def mutation(solutions):
    mutation_prob = random.random()
    if mutation_prob <1:
        solutions = min_x + (max_x - min_x) * random.random()
    return solutions
pop_size = 20
max_gen = 100# 迭代次数#Initialization
min_x=-55
max_x=55
solution=[min_x+(max_x-min_x)*random.random()   for i in range(0,pop_size)]# 随机生成变量

gen_no=0
while gen_no<max_gen:
    function1_values = [function1(solution[i])  for i in range(0,pop_size)]
    function2_values = [function2(solution[i])  for i in range(0,pop_size)]    # 生成两个函数值列表，构成一个种群
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])    # 种群之间进行快速非支配性排序,得到非支配性排序集合
    print("The best front for Generation number ",gen_no, " is")

    for value_1 in non_dominated_sorted_solution[0]:
        print(round(solution[value_1], 3), end=" ")
    print("\n")
    crowding_distance_values=[]    # 计算非支配集合中每个个体的拥挤度
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]

    #生成了子代
    while len(solution2)!=2*pop_size:
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)        # 选择
        solution2.append(crossover(solution[a1],solution[b1]))        #随机选择，将种群中的个体进行交配，得到子代种群2*pop_size
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])    # 将两个目标函数得到的两个种群值value,再进行排序 得到2*pop_size解
    crowding_distance_values2=[]

    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    # 计算子代的个体间的距离值
    new_solution= []

    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]        #排序

        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if len(new_solution)==pop_size:
                break
        if len(new_solution) == pop_size:
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

function1 = [i * -1 for i in function1_values]
function2 = [j * -1 for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()
