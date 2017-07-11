# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/11
from math import log

import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算矩阵数据集的信息熵
def calcShannonEnt(dataSet):
    # 获取矩阵数据集长度
    numEntries = len(dataSet)
    labelCounts = {}
    # 遍历矩阵
    for featVec in dataSet:  # the the number of unique elements and their occurance
        # 获取矩阵当前行的最后一个元素
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            # 如果第一次出现，添加到labelCounts
            labelCounts[currentLabel] = 0
        # 统计当前label出现次数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 遍历label,分别计算所有label出现概率
    for key in labelCounts:
        # 计算当前label出现的概率（p == 出现次数/总数量）
        prob = float(labelCounts[key]) / numEntries
        # 信息熵 (Xi) = -p(xi)*log2(xi)
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


# 返回矩阵数据集上各行第axis位置的值为value的行（不包括axis位置元素）
# dataSet 待划分的数据集
# axis 划分数据集的特征
# value 特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        # 判断是否与value相等
        if featVec[axis] == value:
            # 相等则获取数组[0:axis]特征值
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            # 扩展特征值数组(目的是去掉数组中axis位置元素？？？）
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择一个最好的特征值来分割数据集，返回最好的分割特征
def chooseBestFeatureToSplit(dataSet):
    # 获取特征值数量（最后一个值为label，不是特征值）
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    # 计算基础信息熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    # 遍历矩阵数据集所有特征数据
    for i in range(numFeatures):  # iterate over all the features
        # 遍历获取矩阵数据集的第i列的数据
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        # 去掉重复值
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            # 获取矩阵数据集上各行第位置的值为value的行（不包括位置元素）
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算矩阵数据集中出现value值的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 计算整体信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 获取信息增益
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        # 如果当前信息增益大于最好的信息增益，更新，并保持当前遍历的i值
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


# 返回classList特征向量表中，数量最多的特征
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# dataSet 数据集
# labels 标签列表（包含了数据集中所有特征的标签）
def createTree(dataSet, labels):
    # 获取矩阵数据集最后一列（最后一列为判断fish yes/no）
    # classList包含了所有数据集类标签
    classList = [example[-1] for example in dataSet]
    # 判断所有数据分类都是classList[0]同一分类（即，只有yes或者no一个分类）
    if classList.count(classList[0]) == len(classList):
        # 所有分类相同，则递归结束，返回当前分类
        return classList[0]  # stop splitting when all of the classes are equal
    # 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组
    # 如果数据集已经处理了所有属性，但是类标签依然不是唯一的
    # 此时通常会选用多数表决的方式决定该叶子节点分类
    # dataSet[0]为矩阵数据集第0行
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        # 返回当前类标签数量最多的分类
        return majorityCnt(classList)
    # 获取信息增益最大的特征值
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 定义返回的决策树
    myTree = {bestFeatLabel: {}}
    # 删除labels特征标签中的最佳特征值
    del (labels[bestFeat])
    # 获取所有数据集中最佳特征值列数据
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 全新复制labels
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        # 递归创建决策树
        # splitDataSet返回矩阵数据集上各行第bestFeat位置的值为value的行（不包括bestFeat位置元素），即根据最佳特征分割后的子数据集
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
