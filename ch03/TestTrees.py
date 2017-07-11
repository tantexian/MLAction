# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/11
import trees
import unittest


class TestTrees(unittest.TestCase):
    def test_shannon(self):
        dataSet, labels = trees.createDataSet()
        print ("\n dataSet == %s" % (dataSet))
        shannon = trees.calcShannonEnt(dataSet)
        print ("\n shannon == %s" % (shannon))

        # 不确定性越大，熵越大
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'yes?no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        shannon = trees.calcShannonEnt(dataSet)
        print ("\n shannon == %s" % (shannon))

    def test_array(self):
        dataSet = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'yes?no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        print ("\n dataSet == %s" % (dataSet))
        print ("\n dataSet[0] == %s" % (dataSet[0]))
        axis = 2
        a1 = dataSet[:axis]
        print ("\n a1 before == %s" % (a1))

        a1.extend(dataSet[axis + 1:])
        print ("\n a1 after == %s" % (a1))

    def test_split(self):
        dataSet, labels = trees.createDataSet()
        print ("\n dataSet == %s" % (dataSet))
        dataset1 = trees.splitDataSet(dataSet, 0, 1)
        print ("\n dataSet1 == %s" % (dataset1))
        dataset2 = trees.splitDataSet(dataSet, 0, 0)
        print ("\n dataSet2 == %s" % (dataset2))

    def test_forin(self):
        dataSet, label = trees.createDataSet()
        print ("\n dataSet == %s" % (dataSet))
        featList = [example[2] for example in dataSet]
        print ("\n featList == %s" % (featList))

    # 代码输出结果告诉第0个特征是最好的用于划分数据集的特征
    # 输出为0（no surfacing），表示第一个特征信息增益最大
    def test_bestChoose(self):
        dataSet, label = trees.createDataSet()
        print ("\n dataSet == %s" % (dataSet))
        bestFeature = trees.chooseBestFeatureToSplit(dataSet)
        print ("\n bestFeature == %s" % (bestFeature))

    def test_createTree(self):
        dataSet, labels = trees.createDataSet()
        print ("\n dataSet == %s" % (dataSet))
        tree = trees.createTree(dataSet, labels)
        print("\n tree == %s" % (tree))
