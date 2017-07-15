# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/14
import unittest
import simplesmo
import optsmo
from numpy import *


class smoTest(unittest.TestCase):
    # Ran 1 test in 11.547s
    def test_simple_smo(self):
        data_set, label_set = simplesmo.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = simplesmo.smoSimple(data_set, label_set, 0.6, 0.001, 40)
        print("\n b == %s" % (b))
        print("\n alpha == %s" % (alpha[alpha > 0]))

        for i in range(100):
            if alpha[i] > 0.0:
                print("\n  data_set[i] == %s label_set[i] == %s " % (data_set[i], label_set[i]))

    # Ran 1 test in 0.235s(效率提高了很多)
    def test_opt_smo(self):
        data_set, label_set = simplesmo.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = optsmo.smoOpt(data_set, label_set, 0.6, 0.001, 40)
        print("\n b == %s" % (b))
        print("\n alpha == %s" % (alpha[alpha > 0]))

        for i in range(100):
            if alpha[i] > 0.0:
                print("\n  data_set[i] == %s label_set[i] == %s " % (data_set[i], label_set[i]))

    def test_simple_smo_plot(self):
        data_set, label_set = simplesmo.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = simplesmo.smoSimple(data_set, label_set, 0.6, 0.001, 40)
        ws = simplesmo.calcWs(alpha, data_set, label_set)
        # 其中红色点为支持向量
        simplesmo.plot_smo(data_set, label_set, ws, b, alpha)

    def test_opt_smo_plot(self):
        data_set, label_set = simplesmo.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = optsmo.smoOpt(data_set, label_set, 0.6, 0.001, 40)
        ws = simplesmo.calcWs(alpha, data_set, label_set)
        # 其中红色点为支持向量
        simplesmo.plot_smo(data_set, label_set, ws, b, alpha)

    def test_classify(self):
        data_set, label_set = simplesmo.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = optsmo.smoOpt(data_set, label_set, 0.6, 0.001, 40)
        ws = simplesmo.calcWs(alpha, data_set, label_set)

        # 计算data_set中的第0行的样本数据，来预测分类
        i = 0
        data0 = mat(data_set)[i]
        # 预测值大于0则为1，小于0为-1
        y = data0 * ws + b
        print("\n 根据样本第%s行数据 x == %s \n 得到预测值 == %s \n 真实分类值为 == %s" % (i, data0, y, label_set[i]))

        # 计算data_set中的第5行的样本数据，来预测分类
        i = 5
        data0 = mat(data_set)[i]
        y = data0 * ws + b
        print("\n 根据样本第%s行数据 x == %s \n 得到预测值 == %s \n 真实分类值为 == %s" % (i, data0, y, label_set[i]))

        # 计算data_set中的第5行的样本数据，来预测分类
        i = 8
        data0 = mat(data_set)[i]
        y = data0 * ws + b
        print("\n 根据样本第%s行数据 x == %s \n 得到预测值 == %s \n 真实分类值为 == %s" % (i, data0, y, label_set[i]))
