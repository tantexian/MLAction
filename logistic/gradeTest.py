# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/13
import unittest

from numpy import *

import logRegres


class gradeTest(unittest.TestCase):
    # 测试梯度算法
    def test_grade_plot(self):
        data_set, label_mat = logRegres.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        print ("\n label_mat == %s" % (label_mat))
        weights = logRegres.gradAscent(data_set, label_mat)
        print ("\n weights == %s" % (weights))
        # getA 为将numpy中的矩阵转换为python的array
        logRegres.plotBestFit(weights.getA())

    # 测试随机梯度算法
    def test_stoc_grade_plot(self):
        data_set, label_mat = logRegres.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        print ("\n label_mat == %s" % (label_mat))
        weights = logRegres.stocGradAscent0(array(data_set), label_mat)
        print ("\n weights == %s" % (weights))
        logRegres.plotBestFit(weights)

    # 测试优化后随机梯度算法（动态变化步长）
    def test_best_stoc_grade_plot(self):
        data_set, label_mat = logRegres.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        print ("\n label_mat == %s" % (label_mat))
        # 迭代150次
        weights = logRegres.stocGradAscent1(array(data_set), label_mat, 200)
        print ("\n weights == %s" % (weights))
        # getA 为将numpy中的矩阵转换为python的array
        logRegres.plotBestFit(weights)
