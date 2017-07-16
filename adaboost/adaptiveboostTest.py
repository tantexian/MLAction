# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/16
import unittest
import adaptiveboost
from numpy import *


class adaptiveboostTest(unittest.TestCase):
    # 测试建立单层决策树
    def test_buildstump(self):
        simp_data, label_data = adaptiveboost.loadSimpData()
        # 初始化权重，取平均值，均为0.2
        D = mat(ones((5, 1)) / 5)
        bestStump, minError, bestClasEst  = adaptiveboost.buildStump(simp_data, label_data, D)
        print("\n bestStump == %s" %(bestStump))
        print("\n minError == %s" %(minError))
        print("\n bestClasEst == %s" %(bestClasEst))


    def test_adaboost(self):
        simp_data, label_data = adaptiveboost.loadSimpData()
        adaptiveboost.adaBoostTrainDS(simp_data, label_data, 10)

    def test_lt_gt(self):
        datMat = matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
        classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
        retArray = ones((shape(datMat)[0], 1))
        print("\n retArray == %s" % (retArray))

        i = 1
        rangeMin = datMat[:, i].min()
        print("\n rangeMin == %s" % (rangeMin))
        threshVal = rangeMin + 1
        print("\n threshVal == %s" % (threshVal))
        result = datMat[:, i] <= threshVal
        print("\n result == %s" % (result))

        # retArray[datMat[:, i] <= threshVal] = -1.0
        # print("\n retArray == %s" %(retArray))


        testMat = matrix([[False], [True], [True], [False], [True]])
        # testMat = matrix([[1], [1], [0], [0], [1]])
        # testMat = matrix([[1], [2], [3], [-4], [-1]])
        retArray[testMat] = -1.0
        print("\n retArray == %s" % (retArray))

    def test_multipy(self):
        # 其中值为1表示错误
        errMat = matrix([[1],
                         [0],
                         [1],
                         [0],
                         [1]])
        print("\n errMat == %s" % (errMat))

        D = mat(ones((5, 1)) / 5)
        print("\n D == %s" % (D))
        print("\n D.T == %s" % (D.T))

        # D.T为1*5行向量，errMat为5*3向量
        weightedError = D.T * errMat
        print("\n weightedError type == %s" % (type(weightedError)))

        minErr = 1.5

        if weightedError < minErr:
            print("\n weightedError(%s) < minErr(%s)" % (weightedError, minErr))
        else:
            print("\n weightedError(%s) >= minErr(%s)" % (weightedError, minErr))

        minErr = 0.5

        if weightedError < minErr:
            print("\n weightedError(%s) < minErr(%s)" % (weightedError, minErr))
        else:
            print("\n weightedError(%s) >= minErr(%s)" % (weightedError, minErr))

        minErr = mat([[1.0]])

        if weightedError < minErr:
            print("\n weightedError(%s) < minErr(%s)" % (weightedError, minErr))
        else:
            print("\n weightedError(%s) >= minErr(%s)" % (weightedError, minErr))
