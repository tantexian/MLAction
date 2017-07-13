# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
import unittest

from numpy import *

import logRegres


class logRegresTest(unittest.TestCase):
    def test_grade(self):
        data_set, label_mat = logRegres.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        print ("\n label_mat == %s" % (label_mat))
        ascent = logRegres.gradAscent(data_set, label_mat)
        print ("\n ascent == %s" % (ascent))

    # 自定义样本数据，测试grade回归
    def test_debug_logregres(self):
        # 分类直线为经过（0，1）（3，4）两个点，则直线为y=x+1
        # 其中[x,y,label]分别表示x坐标，y坐标
        # label==1表示在直线之上 label==0表示在直线之下
        # 由于此处数据数值偏大，因此gradAscent方法中的步长也对应应该变大，否则分类线则不太准确
        data_set = [[1, 3, 1],
                    [2, 4, 1],
                    [2, 5, 1],
                    # [-1, 2, 1],
                    # [-2, 4, 1],
                    # [-3, 3, 1],
                    [1, 1, 0],
                    [2, 1, 0],
                    [3, 3, 0]
                    # [0, 0, 0],
                    # [-2, -3, 0],
                    # [-4, -5, 0]
                    ]
        data_set = mat(data_set)
        m, n = shape(data_set)

        print ("\n data_set == %s" % (data_set))

        # 获取出去label的子矩阵
        data_in = data_set[:, :n - 1]
        # 初始化设置x0=1?
        one = ones((m, 1))
        print ("\n one == %s" % (one))
        data_in = column_stack((one, data_in))

        classmat = data_set[:, n - 1:]
        classLabel = []
        for i in classmat.flat:
            classLabel.append(i)

        print ("\n\n data_in == %s" % (data_in))
        print ("\n classLabel == %s" % (classLabel))
        weights = logRegres.gradAscent(data_in, classLabel)
        print ("\n weights == %s" % (weights))
        # getA 为将numpy中的矩阵转换为python的array
        logRegres.plotBestFit1(weights.getA(), data_in, classLabel)
