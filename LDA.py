#coding=utf-8
'''
***** LDA *****
author : zsz
email  : zhengsz@pku.edu.cn
last modify: 2017-12-8
description: a simple impletation of LDA algorithm
copyright: the codes following are free, hope these codes can be helpful
***** ******** *****
'''
__author__ = 'zsz'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# this is a module written to get data
import fetchData

# the filename is the name of the file in which data is stored
def LDA(filename):
    # get the data, the usage of fetchData function can be found in the README file
    xData, yData = fetchData.fetchData(filename=filename,x_beg=1,x_size=3,y_beg=0,y_size=1)
    # we need to use yData as keys in dict, so it's nice to turn it to list
    yData = list(yData.reshape(-1))
    N = len(xData)
    newData = {}
    for i in range(N):
        # you can see, dicts are really easy to use
        if newData.get(yData[i]) == None:
            newData[yData[i]] = []
        newData[yData[i]].append(xData[i, :])
    for p in newData:
        newData[p] = np.mat(newData[p])
    M = {}
    L = {}
    for p in newData:
        # again! how easily! we even do not need to know what the labels are!
        M[p] = np.mean(newData[p], 0)
        L[p] = len(newData[p])
    m = np.mean(xData, 0)
    SW, SB = {}, {}
    for p in newData:
        SW[p] = np.cov(newData[p] - M[p],rowvar=False)
        diff = M[p] - m
        SB[p] = L[p] * (diff.T * diff)
    Sb, Sw = None, None
    # the following seemingly foolish codes are intended to initialize Sb and Sw
    for p in newData:
        Sb = np.zeros_like(SB[p])
        Sw = np.zeros_like(SW[p])
        break
    for p in newData:
        Sb = Sb + SB[p]
        Sw = Sw + SW[p]

    # compute the eigenvalues and right vectors
    v, w = np.linalg.eig(np.linalg.inv(Sw) * Sb)
    # this sort operation can make things from small to big, although we just take the index
    v_index = np.argsort(v)
    # let's get the top two
    w_index = v_index[-1:-3:-1]
    w2 = w[:,w_index]

    # now it's time for drawing!
    color = ['g', 'b', 'y']
    count = 0
    ax = plt.subplot('121', projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('initial data')
    for p in newData:
        # we can use the convenience of dicts again
        x1 = newData[p][:,0]
        x1 = np.array(x1).reshape(-1)
        x2 = newData[p][:,1]
        x2 = np.array(x2).reshape(-1)
        x3 = newData[p][:,2]
        x3 = np.array(x3).reshape(-1)
        ax.scatter(x1,x2,x3,color=color[count])
        count += 1
    # show the main directions
    line = np.linspace(-10,10,1000)
    l1 = w2[0,0] * line + m[0]
    l2 = w2[1,0] * line + m[1]
    l3 = w2[2,0] * line + m[2]
    ax.plot(l1, l2, l3, color='pink', label='first')
    l1 = w2[0, 1] * line + m[0]
    l2 = w2[1, 1] * line + m[1]
    l3 = w2[2, 1] * line + m[2]
    ax.plot(l1, l2, l3, color='purple', label='second')

    # get new representations of data
    projectX = {}
    count = 0
    plt.subplot('122')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('LDA result 2-d')
    for p in newData:
        projectX[p] = newData[p] * w2
        x1 = projectX[p][:,0]
        x1 = np.array(x1).reshape(-1)
        x2 = projectX[p][:,1]
        x2 = np.array(x2).reshape(-1)
        plt.scatter(x1, x2, c=color[count])
        count += 1
    plt.show()


if __name__ == '__main__':
    LDA('sample.txt')