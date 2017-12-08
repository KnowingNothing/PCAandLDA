#coding=utf-8
'''
***** fetchData *****
author : zsz
email  : zhengsz@pku.edu.cn
last modify: 2017-12-8
description: a simple tool to handle file operations
copyright: the codes following are free, hope these codes can be helpful
***** ******** *****
'''
__author__ = 'zsz'
import numpy as np

# x refers to the data and y refers to the label
def fetchData(filename, x_beg, x_size, y_beg, y_size):
    raw = np.loadtxt(filename)
    x = raw[:, x_beg:x_beg + x_size]
    y = raw[:, y_beg:y_beg + y_size]
    return x, y
