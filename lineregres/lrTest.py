# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17
import unittest

import matplotlib.pyplot as plt
from numpy import *

import lr
import lwlr


class lrTest(unittest.TestCase):
    def test_lr(self):
        dataMat, labelMat = lr.loadDataSet("ex0.txt")
        w = lr.standRegres(dataMat, labelMat)
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

    def test_corrcoef(self):
        dataMat, labelMat = lr.loadDataSet("ex0.txt")
        w = lr.standRegres(dataMat, labelMat)
        print ("\n w == %s" % (w))

        xMat = mat(dataMat)
        yMat = mat(labelMat)
        # 预测值
        yHat = xMat * w
        # 计算预估值与真实值的相关系数
        n = corrcoef(yHat.T, yMat)
        print ("\n corrcoef == %s" % (n))
        # 因为yMat和自己的匹配是最完美的，因此对角线为1
        # [[ 1.          0.98647356]
        #  [ 0.98647356  1.        ]]

    def test_lwlr(self):
        dataMat, labelMat = lr.loadDataSet("ex0.txt")
        # 对dataMat第一行的样本数据进行预测
        w = lwlr.lwlr(dataMat[0], dataMat, labelMat, 1)
        print ("\n w == %s" % (w))

        yHat = lwlr.lwlrTest(dataMat, dataMat, labelMat, 1)
        print ("\n yHat == %s" % (yHat))

    def test_lwlr_plot(self):
        dataMat, labelMat = lr.loadDataSet("ex0.txt")
        # k = 1 则与线性回归基本一致（当k=0.01时，预测结果比较好）
        lwlr.lwlrPlot(dataMat, labelMat, 0.01)

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
