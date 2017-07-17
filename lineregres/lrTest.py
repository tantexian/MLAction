# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17
import unittest

import matplotlib.pyplot as plt
from numpy import *

import regression


class lrTest(unittest.TestCase):
    def test_lr(self):
        dataMat, labelMat = regression.loadDataSet("ex0.txt")
        w = regression.standRegres(dataMat, labelMat)
        print ("\n w == %s" % (w))

        xMat = mat(dataMat)
        yMat = mat(labelMat)
        # 预测值
        yHat = xMat * w

        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

        xCopy = xMat.copy()
        # 进行排序
        xCopy.sort(0)
        yHat1 = xCopy * w
        print ("\n xCopy[:, 1] == %s" % (xCopy[:, 1]))
        x = xCopy[:, 1]
        y = yHat1
        # 画直线
        ax.plot(x.A, y.A, "r")
        plt.show()

    def test_Mat_T(self):
        x = arange(4).reshape(2, 2)
        print ("\n x == %s" % (x))

        # xI = x.I
        # print ("\n xI == %s" % (xI))

        xT = x.T
        print ("\n xT == %s" % (xT))

        xT1 = x.transpose()
        print ("\n xT1 == %s" % (xT1))

        xTx = x.T * x
        print ("\n xTx == %s" % (xTx))
