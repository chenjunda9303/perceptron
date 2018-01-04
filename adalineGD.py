# -*- coding:utf-8 -*-
import numpy as np
import perceptron


class AdalineGD(object):
    def __int__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        
        :param X: 二维数组[n_sample,n_features]
        n_sample:X中含有训练数据条目数
        n_features:含有4个数据的一位向量，用于表示一条训练条目
        
        :param y:一维向量，存储每一条训练条目的正确分类 
        :return: 
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = perceptron.Perceptron.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() /2.0
            self.cost_.append(cost)

    def activation(self, X):
        return perceptron.Perceptron.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)
