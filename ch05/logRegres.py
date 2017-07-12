# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
from numpy import *


# 返回数据矩阵及label矩阵向量
# 从testSet.txt文本中加载数据
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid公式计算（y = 1/(1+1/e^x)）
# 注意：此处inX可以是行向量
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 使用梯度上升方式或者最佳回归权重参数
# dataMatIn 2维的numpy数组，每列代表不同的特征，每行代表一个训练数据
# classLabels 类标签，1*100 的行向量
def gradAscent(dataMatIn, classLabels):
    # 转换为numpy的矩阵
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    # 将行向量，转换为numpy的列向量
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    # 获取数组的行列
    m, n = shape(dataMatrix)
    # 步长设置为0.001
    alpha = 0.001
    # 最大迭代次数设置为500
    maxCycles = 500
    # 构建n*1全1矩阵
    weights = ones((n, 1))
    # 迭代maxCycles次
    for k in range(maxCycles):  # heavy on matrix operations
        # 使用s函数求值（矩阵相乘:左行乘右列）
        # TODO: 此处逻辑？？？
        h = sigmoid(dataMatrix * weights)  # matrix mult
        # 向量相减（h也为列向量）
        error = (labelMat - h)  # vector subtraction
        # transpose矩阵转秩
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()
