# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17
import unittest

import abalone
import lr
import lwlr


class abaloneTest(unittest.TestCase):
    def test_abalone(self):
        data_set, label_set = lr.loadDataSet("abalone.txt")
        train_data_set = data_set[0:99]
        train_label_set = label_set[0:99]
        yHat01 = lwlr.lwlrTest(train_data_set, train_data_set, train_label_set, 0.1)
        yHat1 = lwlr.lwlrTest(train_data_set, train_data_set, train_label_set, 1)
        yHat10 = lwlr.lwlrTest(train_data_set, train_data_set, train_label_set, 10)
        # 计算不同k值的误差值：
        error01 = abalone.rssError(train_label_set, yHat01.T)
        # error01 == 56.7987246777
        print ("\n error01 == %s" % (error01))

        error1 = abalone.rssError(train_label_set, yHat1.T)
        # error1 == 429.89056187
        print ("\n error1 == %s" % (error1))

        error10 = abalone.rssError(train_label_set, yHat10.T)
        # error10 == 549.118170883
        print ("\n error10 == %s" % (error10))
        # 由上述结果发现，使用最小的核 k=0.1 得到最低的误差。但是对于新数据则不然。


        new_data_set = data_set[100:199]
        new_label_set = label_set[100:199]
        # 使用全新数据计算不同k值的误差值：(发现k=10的误差最小)
        yHat01 = lwlr.lwlrTest(new_data_set, train_data_set, train_label_set, 0.1)
        yHat1 = lwlr.lwlrTest(new_data_set, train_data_set, train_label_set, 1)
        yHat10 = lwlr.lwlrTest(new_data_set, train_data_set, train_label_set, 10)
        # 计算不同k值的误差值：
        error01 = abalone.rssError(new_label_set, yHat01.T)
        # error01 == 56.7987246777
        print ("\n\n\n error01 == %s" % (error01))

        error1 = abalone.rssError(new_label_set, yHat1.T)
        # error1 == 429.89056187
        print ("\n error1 == %s" % (error1))

        error10 = abalone.rssError(new_label_set, yHat10.T)
        # error10 == 549.118170883
        print ("\n error10 == %s" % (error10))
