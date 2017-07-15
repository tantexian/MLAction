# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/11

import operator
from os import listdir

from numpy import *


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
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 得到数组的行数
    # 变成和dataSet一样的行数,行数=原来*numSamples，列数=原来*1 ，然后每个特征点和样本的点进行相减
    # tile(a,(b,c)),在行方向将数组a重复b次，在列方向将数组a重复c次
    # diffMat 为将所有节点的x，y与当前inX做减法等到x,y轴坐标差值
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # axis=1为按行求和(0 为按列求和)表示（x^2 + y^2）
    sqDistances = sqDiffMat.sum(axis=1)
    # 平方根
    distances = sqDistances ** 0.5

    # 排序，argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    # print ("sortedDistIndicies == %s" % (sortedDistIndicies))
    classCount = {}
    # 取排序后前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 返回前K个点出现频率最高的类别作为当前点的预测分类
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    # 打开文件
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # get the number of lines in the file
    # 准备一个numberOfLines*3的全零的矩阵
    returnMat = zeros((numberOfLines, 3))  # prepare matrix to return
    classLabelVector = []  # prepare labels return
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        # 逐行赋值到矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 如果数组最后一个值为数组，则保存到label数组中
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:  # 否则根据label类型对应的树枝保存起来
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 由于每个特征值的度量不一致（飞行公里1000+，而消耗冰激凌为0.1+）
# 而每个特征值同等重要，因此为了消除度量不一致，需要归一化数值（0~1或者-1~1区间）
def autoNorm(dataSet):
    # 还去数据集中的最小，最大值，及区间范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    # shape为根据dateset矩阵返回矩阵的(行数，列数)
    normDataSet = zeros(shape(dataSet))
    # 获取矩阵的行数
    m = dataSet.shape[0]
    # 计算出矩阵每一项和最小值的差值
    normDataSet = dataSet - tile(minVals, (m, 1))  # tile 重复构造milVals m行，1列
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # normMat[i, :]表示获取第i行数据
        # normMat[numTestVecs:m, :] 表示numTestVecs:m行的数据作为样本数据
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print
        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print
    "the total error rate is: %f" % (errorCount / float(numTestVecs))
    print
    errorCount


# 输入person参数，通过k近邻方法根据已有样本，判断出当前person的类型
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print
    "You will probably like this person: %s" % resultList[classifierResult - 1]


# 将filename对应的数字特征值保存到特征矩阵中
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 32个列
    for i in range(32):
        lineStr = fr.readline()
        # 每一行32个值
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    # 获取训练数据样本(每个样本数据为32*32=1024 bit)
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    # 有m个数据，则建立m*1024的训练矩阵
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        # 依次将第i个数据文件样本特征转换保存到训练矩阵中
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 用testFile的值来进行测试
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print
        "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print
    "\nthe total number of errors is: %d" % errorCount
    print
    "\nthe total error rate is: %f" % (errorCount / float(mTest))
