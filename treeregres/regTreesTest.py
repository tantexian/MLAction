# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
import unittest

from numpy import *

import linearModelTrees
import pruneTrees
import regTrees


class regTreesTest(unittest.TestCase):
    def test_reg_trees(self):
        # regTrees.loadDataSet()
        # 创建对角矩阵
        testMat = mat(eye(4))
        print ("\n testMat == %s" % (testMat))
        # 将第1列的特征值根据阈值0.5分割成两个字矩阵
        mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
        print ("\n mat0 == %s" % (mat0))
        print ("\n mat1 == %s" % (mat1))

    def test_create_reg_tree(self):
        data_set = regTrees.loadDataSet("ex0.txt")
        data_set = mat(data_set)
        print ("\n data_set == %s" % (data_set))
        tree = regTrees.createTree(data_set)
        print ("\n tree == %s" % (tree))

        # 测试过拟合
        tree = regTrees.createTree(data_set, ops=(0, 1))
        print ("\n tree == %s" % (tree))

    def test_prune(self):
        data_set = regTrees.loadDataSet("ex2.txt")
        data_set = mat(data_set)

        tree = regTrees.createTree(data_set, ops=(1, 4))
        print ("\n tree == %s" % (tree))

        test_set = regTrees.loadDataSet("ex2test.txt")
        test_set = mat(test_set)
        # 后剪枝，可以看到没有像预期被剪成两部分，说明后剪枝可能不如先剪枝有效
        pruneTree = pruneTrees.prune(tree, test_set)
        print ("\n pruneTree == %s" % (pruneTree))

    def test_linear_model_tree(self):
        data_set = regTrees.loadDataSet("exp2.txt")
        data_set = mat(data_set)

        tree = regTrees.createTree(data_set, linearModelTrees.modelLeaf, linearModelTrees.modelErr, ops=(1, 10))
        print ("\n tree == %s" % (tree))
