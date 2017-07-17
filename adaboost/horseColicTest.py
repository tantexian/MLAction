# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17
import unittest

from numpy import *

import adaptiveboost
import horseColic


class horseColicTest(unittest.TestCase):
    def test_colic(self):
        data_set, label_set = horseColic.loadDataSet("horseColicTraining2.txt")
        # 改变此处的分类器数目（迭代次数），得到不同的结果（错误率变化）
        weakClassArr, aggClassEst = adaptiveboost.adaBoostTrainDS(data_set, label_set, 50)
        print ("\n weakClassArr == %s" % (weakClassArr))

        data_set, label_test = horseColic.loadDataSet("horseColicTest2.txt")
        # 获取测试数据行数
        m = shape(data_set)[0]
        # 获取预测值
        aggClassEst = adaptiveboost.adaClassify(data_set, weakClassArr)
        print ("\n aggClassEst == %s" % (aggClassEst))

        # 构造一个全1列向量
        errMat = mat(ones((m, 1)))
        # 将预测值与真实值不相等的矩阵元素过滤出来，相加
        err_sum = errMat[aggClassEst != mat(label_test).T].sum()
        err_rate = err_sum / m
        print ("\n err_sum == %s err_rate == %s" % (err_sum, err_rate))

    def test_plot_roc(self):
        data_set, label_set = horseColic.loadDataSet("horseColicTraining2.txt")
        # 改变此处的分类器数目（迭代次数），得到不同的结果（错误率变化）
        weakClassArr, aggClassEst = adaptiveboost.adaBoostTrainDS(data_set, label_set, 10)
        print ("\n weakClassArr == %s" % (weakClassArr))

        horseColic.plotROC(aggClassEst.T, label_set)
