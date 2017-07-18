# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
from numpy import *

import regTrees


# 判断是否为一棵树
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 递归函数，从上到下遍历树直到叶子节点为止。如果找到两个叶子节点则计算它们的平均值。
# 该函数对树进行坍陷处理（即返回🌲平均值），在prune中调用应明确这一点。
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 功能：对已知树进行剪枝
# tree : 待剪枝的树
# testData : 剪枝所需的测试数据
# 返回：xxx
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
def prune(tree, testData):
    # 判断测试集是否为空，为空，直接返回原始树
    if shape(testData)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        # 根据待剪枝树的特征索引及切分阈值，对测试数据进行切分（获取新的切分后的测试数据lSet，rSet）
        lSet, rSet = regTrees.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 如果左节点为树，
    if isTree(tree['left']):
        # 递归对树的左节点进行剪枝
        tree['left'] = prune(tree['left'], lSet)
    # 如果右节点为树
    if isTree(tree['right']):
        # 递归对树的右节点进行剪枝
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    # 如果左右节点都为叶子节点，则进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 根据待剪枝树的特征索引及切分阈值，对测试数据进行切分（获取新的切分后的测试数据lSet，rSet）
        lSet, rSet = regTrees.binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算为合并之前的误差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        # 合并后节点的平均值tree
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并后的误差
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 合并后的误差小于不合并的误差，则返回合并后的平均值tree，否则返回原始tree
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree
