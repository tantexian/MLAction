# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/19

from numpy import *


# 加载数据
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 创建集合C1,所有大小为1的候选项集的集合
# 遍历所有项，存储不重复的项
def createC1(dataSet):
    C1 = []
    # 遍历样本每行数据
    for transaction in dataSet:
        # 遍历每行数据的每个元素
        for item in transaction:
            # 不在已添加列表中，则添加
            if not [item] in C1:
                C1.append([item])

    # 对C1进行排序
    C1.sort()
    # 对C1中的每个项建立一个不变集合
    # map为map-reduce相关，函数式编程（对所有元素C1迭代使用frozenset函数）
    # frozenset代表为不可变set集合
    return map(frozenset, C1)  # use frozen set so we
    # can use it as a key in a dict


# 功能：用于从C1生成L1
# D : 数据集合列表
# Ck : 候选集Ck
# minSupport : 感兴趣项的最小支持度
# 返回：
# retList : 满足支持度的字典元素
# supportData : 最频繁项支持度集合（val为支持度）
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/19
def scanD(D, Ck, minSupport):
    # 创建空字典
    ssCnt = {}
    # 遍历数据集中所有记录
    for tid in D:
        # 遍历Ck中所有候选集
        for can in Ck:
            # 如果Ck中集合是D中一部分，则增加字典中对应计数值
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    # 计算每个元素的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        # 大于等于最小支持度的元素，将字典元素添加到retList列表中
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk, k):  # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2];
            L2 = list(Lk[j])[:k - 2]
            L1.sort();
            L2.sort()
            if L1 == L2:  # if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j])  # set union
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
