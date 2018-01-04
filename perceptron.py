# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# 感知器类
class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量训练次数
    w_:权重向量
    errors_:判断错误的次数
    """
    def __init__(self, eta=0.01, n_iter=10):
        # 私有属性eta、n_iter类外部无法访问
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """
        :param X:输入样本 
        :param y: 对应样本分类
        X:shape{n_samples, n_features}
        X:[[1,2,3],[4,5,6]]
        n_samples=2
        n_features=3
        
        y:[1,-1]
        :return: 
        """

        # 初始化权重向量  np.zeros 生成一组全为0的向量
        # 加1是步调函数的阈值w0
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []


        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):

                # update = n * (y - y')
                update = self.eta * (target - self.predict(xi))

                # xi是一个向量  即常数乘以向量  w_[1:]表示从第1个元素赋值等号后的向量
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            pass
        pass

    def net_input(self,X):
        """
        电信号与神经分叉权重的点积
        Z = w0*1 + w1*x1 + ...+wn*wn
        :param X: 
        :return: 
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

file = 'F:\my_pycode\AdalineGD\data.txt'
df = pd.read_csv(file, header=None)
# print df.head(10)
y = df.loc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
X = df.iloc[0:100, [0, 2]].values
ppn = Perceptron(eta=0.1, n_iter=50)
ppn.fit(X, y)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    market = ("s", "x", "o", "v")

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    print xx1, xx1.ravel()
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    print np.array([xx1.ravel(), xx2.ravel()]).T
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())

    plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="-1")
    plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="1")

plot_decision_regions(X, y, ppn, resolution=0.02)
plt.xlabel("leaf_len")
plt.ylabel("rachis_len")
plt.legend(loc="upper left")
plt.show()




