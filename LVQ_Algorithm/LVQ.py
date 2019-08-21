# -*- coding:utf-8 -*-
import numpy as np
import copy
from sklearn.datasets import make_moons
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

#结果有一定概率结果会出错，不是太熟悉该算法，所以未解决


def dist(x1, x2):
    d = np.linalg.norm(x1 - x2)
    return d


def get_mu(X, Y):
    k = len(set(Y))
    index = np.random.choice(X.shape[0], 1, replace=False)
    mu1 = [X[index]]
    mus_label = [Y[index]]
    for _ in range(k - 1):
        max_dist_index = 0
        max_distance = 0
        for j in range(X.shape[0]):
            min_dist_with_mu = 999999

            for mu in mu1:
                dist_with_mu = dist(mu, X[j])
                if min_dist_with_mu > dist_with_mu:
                    min_dist_with_mu = dist_with_mu

            if max_distance < min_dist_with_mu:
                max_distance = min_dist_with_mu
                max_dist_index = j
        mu1.append(X[max_dist_index])
        mus_label.append(Y[max_dist_index])

    mus_array = np.array([])
    for i in range(k):
        if i == 0:
            mus_array = mu1[i]
        else:
            mu1[i] = mu1[i].reshape(mu1[0].shape)
            mus_array = np.append(mus_array, mu1[i], axis=0)
    mus_label_array = np.array(mus_label)
    return mus_array, mus_label_array


class LVQ:
    def __init__(self, X,Y,max_iter=10000, eta=0.1, e=0.01):
        self.mus_array, self.mus_label_array = get_mu(X, Y)
        self.max_iter = max_iter
        self.eta = eta
        self.e = e

    def get_mu_index(self, x):
        min_dist_with_mu = 999999
        index = -1

        for i in range(self.mus_array.shape[0]):
            dist_with_mu = dist(self.mus_array[i], x)
            if min_dist_with_mu > dist_with_mu:
                min_dist_with_mu = dist_with_mu
                index = i

        return index

    def fit(self, X, Y):
        iterate = 0

        while iterate < self.max_iter:
            old_mus_array = copy.deepcopy(self.mus_array)
            index = np.random.choice(Y.shape[0], 1, replace=False)

            mu_index = self.get_mu_index(X[index])
            if self.mus_label_array[mu_index] == Y[index]:
                self.mus_array[mu_index] = self.mus_array[mu_index] + self.eta * (X[index] - self.mus_array[mu_index])
            else:
                self.mus_array[mu_index] = self.mus_array[mu_index] - self.eta * (X[index] - self.mus_array[mu_index])

            diff = 0
            for i in range(self.mus_array.shape[0]):
                diff += np.linalg.norm(self.mus_array[i] - old_mus_array[i])
            if diff < self.e:
                print('迭代{}次退出'.format(iterate))
                return
            iterate += 1
        print("迭代超过{}次，退出迭代".format(self.max_iter))


if __name__ == '__main__':

    fig = plt.figure(1)

    plt.subplot(221)
    center = [[1, 1], [-1, -1], [1, -1]]
    cluster_std = 0.35
    X1, Y1 = make_blobs(n_samples=500, centers=center,n_features=2, cluster_std=cluster_std, random_state=1)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)

    plt.subplot(222)
    lvq1 = LVQ(X1,Y1)
    lvq1.fit(X1, Y1)
    mus = lvq1.mus_array
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')

    plt.subplot(223)
    X2, Y2 = make_moons(n_samples=500, noise=0.1)
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)

    plt.subplot(224)
    lvq2 = LVQ(X2,Y2)
    lvq2.fit(X2, Y2)
    mus = lvq2.mus_array
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2)
    plt.scatter(mus[:, 0], mus[:, 1], marker='^', c='r')
    plt.show()
