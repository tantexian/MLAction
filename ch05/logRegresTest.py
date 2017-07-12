# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
import unittest

import logRegres


class logRegresTest(unittest.TestCase):
    def test_grade(self):
        data_set, label_mat = logRegres.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        print ("\n label_mat == %s" % (label_mat))
        ascent = logRegres.gradAscent(data_set, label_mat)
        print ("\n ascent == %s" % (ascent))

    def test_plot(self):
        data_set, label_mat = logRegres.loadDataSet()
        weights = logRegres.gradAscent(data_set, label_mat)
        print ("\n weights == %s" % (weights))
        # getA 为将numpy中的矩阵转换为python的array
        logRegres.plotBestFit(weights.getA())
