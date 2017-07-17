# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17
# 局部加权线性回归
import matplotlib.pyplot as plt
from numpy import *


# 功能：根据已有样本数据预测testPoint测试数据的label值
# testPoint : 一行样本数据
# xArr : 训练样本数据
# yArr : 训练样本数据的label
# k : 控制衰减速度参数
# 返回：xxx
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    # 创建对角矩阵（为每个样本初始化一个矩阵）
    weights = mat(eye((m)))
    # 遍历样本数据集，计算每个样本对应的权重（随着样本与待预测点的距离递增，权重以指数级衰减）
    for j in range(m):  # next 2 lines create weights matrix
        diffMat = testPoint - xMat[j, :]  #
        # 权重大小以指数级衰减（k控制衰减速度参数）
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    # 判断矩阵是否可逆
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


#
def lwlrTest(testArr, xArr, yArr, k=1.0):  # loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    # 遍历所有样本数据，依次计算每一行数据样本的预测值
    for i in range(m):
        # 根据已有样本数据xArr及yArr预测testPoint测试数据的label值
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def lwlrPlot(xArr, yArr, k=1.0):  # same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))  # easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)

    xMat = mat(xArr)
    yMat = mat(yArr)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    # 画直线
    ax.plot(xCopy[:, 1].A, yHat, "r")
    plt.show()

    return yHat, xCopy
