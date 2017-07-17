# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17

from numpy import *


# 加载样本数据
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1  # get number of fields
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 计算最佳拟合直线（最小二乘法即均方差）
def standRegres(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 判断行列式是否为0，如果为0，则不存在逆矩阵
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    # w = (xTx)^-1*xT*y
    # 其中xTx.I表示为(xTx)^-1即xTx的逆矩阵
    # numpy提供一个函数方法来计算：
    # ws = linalg.solve(xTx, xMat.T * yMat)
    ws = xTx.I * (xMat.T * yMat)
    return ws
