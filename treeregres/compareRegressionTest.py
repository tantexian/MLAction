# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
import unittest

from numpy import *

import compareRegression
import linearModelTrees
import regTrees


class test(unittest.TestCase):
    def test_compare(self):
        train_mat = mat(regTrees.loadDataSet("bikeSpeedVsIq_train.txt"))
        test_mat = mat(regTrees.loadDataSet("bikeSpeedVsIq_test.txt"))
        # 创建一个回归树
        myTree = regTrees.createTree(train_mat, ops=(1, 20))
        yHat = compareRegression.createForeCast(myTree, test_mat[:, 0])
        corrcoef = compareRegression.corrcoef(yHat, test_mat[:, 1], rowvar=0)[0, 1]
        print ("\n corrcoef == %s" % (corrcoef))

        # 创建一颗模型树
        myTree = regTrees.createTree(train_mat, linearModelTrees.modelLeaf, linearModelTrees.modelErr, ops=(1, 20))
        yHat = compareRegression.createForeCast(myTree, test_mat[:, 0], compareRegression.modelTreeEval)
        corrcoef = compareRegression.corrcoef(yHat, test_mat[:, 1], rowvar=0)[0, 1]
        print ("\n corrcoef == %s" % (corrcoef))

        # 标准回归
        ws, X, Y = linearModelTrees.linearSolve(train_mat)
        print ("\n ws == %s" % (ws))
        for i in range(shape(test_mat)[0]):
            yHat[i] = test_mat[i, 0] * ws[1, 0] + ws[0, 0]
        corrcoef = compareRegression.corrcoef(yHat, test_mat[:, 1], rowvar=0)[0, 1]
        print ("\n corrcoef == %s" % (corrcoef))

