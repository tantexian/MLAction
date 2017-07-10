# coding=utf-8
import operator
from numpy import array, tile


# 创建数据集
def createDateSet():
    # 我们将数据点(1，1.1)定义为A类，数据点(0, 0.1)定义为B类。
    # 然后使用k近邻方法求出1.0, 1.0], [0, 0]属于哪个类
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# intX 分类的输人向量
# dataSet 输人的训练样本集
# labels 标签向量
# k 表示用于选择最近邻居的数目
def classifyO(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
