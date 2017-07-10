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


# intX 分类的输人向量 【0，0】
# dataSet 输人的训练样本集（上述group）
# labels 标签向量 （labels）
# k 表示用于选择最近邻居的数目 （k==3）
def classifyO(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 得到数组的行数
    # 变成和dataSet一样的行数,行数=原来*numSamples，列数=原来*1 ，然后每个特征点和样本的点进行相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # axis=1为按行求和(0 为按列求和)
    sqDistances = sqDiffMat.sum(axis=1)
    # 平方根
    distances = sqDistances ** 0.5
    # 排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 取排序后前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 返回前K个点出现频率最高的类别作为当前点的预测分类
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
