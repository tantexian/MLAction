# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
from numpy import *


def regTreeEval(model, inDat):
    return float(model)


def linearSolve(dataSet):  # helper function used in two places
    m, n = shape(dataSet)
    # 将x，y数据格式化
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    # 判断是否可逆
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    # 使用线性回归公式获取ws
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 当数据不需要再切分时，负责生成叶节点模型，返回线性模型系数ws
def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


# 给定数据集上计算误差（总方差）
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))
