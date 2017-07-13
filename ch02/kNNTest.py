# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/11

import unittest

import matplotlib.pyplot as plt
from numpy import *

import kNN


class kNNTest(unittest.TestCase):
    def test_create(self):
        group, labels = kNN.createDateSet()
        print("group == %s\n labels == %s" % (group, labels))
        return

    def test_classify0(self):
        group, labels = kNN.createDateSet()
        # 根据输入的【0，0】判别出属于哪种类型（labels）
        result = kNN.classifyO([0, 0], group, labels, 3)
        print("result == %s" % (result))

        result = kNN.classifyO([1, 1], group, labels, 3)
        print("result == %s" % (result))

    def test_file2mat(self):
        fileName = "datingTestSet.txt"
        matrix, label = kNN.file2matrix(fileName)
        print("\nmatrix == %s" % (matrix))
        print("\nlabel == %s" % (label))

    def test_matplot(self):
        fileName = "datingTestSet.txt"
        datingDataMat, datingLabels = kNN.file2matrix(fileName)
        # 创建一幅图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 散点图使用datingDataMat矩阵的第二、第三列数据，
        # 分别表示特征值 横轴表示“玩视频游戏所耗时间百分比”
        # 纵轴表示“每周所消费的冰淇淋公升数” 。
        # datingDataMat[:, 1] 表示矩阵中所有行中第一列的数据
        ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
        plt.show()

    def test_autoNorm(self):
        fileName = "datingTestSet.txt"
        datingDataMat, datingLabels = kNN.file2matrix(fileName)
        print ("\n datingDataMat == %s" % (datingDataMat))
        normDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)
        print("\n normDataSet == %s \n ranges == %s \n minVals == %s \n" % (normDataSet, ranges, minVals))

    def test_acquire(self):
        kNN.datingClassTest()

    def test_classifyPerson(self):
        kNN.classifyPerson()

    def test_handwriting(self):
        kNN.handwritingClassTest()


if __name__ == '__main__':
    unittest.main()
