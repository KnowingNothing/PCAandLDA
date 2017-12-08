#coding=utf-8
'''
***** PCA *****
author : zsz
email  : zhengsz@pku.edu.cn
last modify: 2017-12-8
description: a simple impletation of PCA algorithm
copyright: the codes following are free, hope these codes can be helpful
***** ******** *****
'''
__author__ = 'zsz'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# this is a module written to get data
import fetchData

# without much verbose codes, starts directly
if __name__ == '__main__':
    # get the data, the usage of fetchData function can be found in the README file
    xData, yData = fetchData.fetchData('sample.txt', x_beg=1, x_size=3, y_beg=0, y_size=1)
    # change data to matrix for convenience
    xData = np.mat(xData)
    # it is important to subtracts the mean value
    xData = xData - np.mean(xData, 0)
    # AA can be also computed by 'AA = xData.T * xData'
    AA = np.cov(xData, rowvar=False)
    # the convenient function to get eigenvalue and the right vectors
    v, w = np.linalg.eig(AA)
    # to know which ones are bigger eigenvalues
    v_index = np.argsort(v)
    # to take the top 2
    w_index = v_index[-1:-3:-1]
    w2 = w[:,w_index]
    # to get the new representation of data
    projectX = xData * w2

    # draw some pictures for joy
    plt.subplot('122')
    x1 = projectX[:, 0]
    x1 = np.array(x1).reshape(-1)
    x2 = projectX[:, 1]
    x2 = np.array(x2).reshape(-1)
    plt.scatter(x1, x2, c='b')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('PCA results 2-d')

    ax = plt.subplot('121', projection='3d')
    x1 = xData[:, 0]
    x1 = np.array(x1).reshape(-1)
    x2 = xData[:, 1]
    x2 = np.array(x2).reshape(-1)
    x3 = xData[:, 2]
    x3 = np.array(x3).reshape(-1)
    ax.scatter(x1, x2, x3, color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('initial data')

    # to draw the main directions
    mainW = np.array(w2[:,0]).reshape(-1)
    line = np.linspace(-10,10,1000)
    l1 = mainW[0] * line
    l2 = mainW[1] * line
    l3 = mainW[2] * line
    ax.plot(l1, l2, l3, color='yellow')
    secondW = np.array(w2[:, 1]).reshape(-1)
    l1 = secondW[0] * line
    l2 = secondW[1] * line
    l3 = secondW[2] * line
    ax.plot(l1, l2, l3, color='green')

    plt.show()