# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/20

from numpy import *


# 加载样本数据，根据tab键分割
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


# 功能：PAC主成分分析
# dataMat : 进行pac分析的样本数据
# topNfeat : 可选参数，应用的N个特征。，不指定则返回前9999999个特征
# 返回：
# lowDDataMat : 降维后的数据集
# reconMat : 针对原始数据进行重构后的数据集，用于调试
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/20
def pca(dataMat, topNfeat=9999999):
    # 对样本矩阵每一列特征值求平均值
    meanVals = mean(dataMat, axis=0)
    # 每一行的值，分别减去对应的mean行矩阵值（获取到的为每个特征值与特征值平均值的偏差）
    meanRemoved = dataMat - meanVals  # remove mean
    # 求协方差矩阵（更多请参考博文：https://my.oschina.net/tantexian/blog/1476444）
    covMat = cov(meanRemoved, rowvar=0)
    # 求特征值和特征向量（更多请参考博文：https://my.oschina.net/tantexian/blog/1476446）
    eigVals, eigVects = linalg.eig(mat(covMat))
    # 对特征值从小到大排序
    eigValInd = argsort(eigVals)  # sort, sort goes smallest to largest
    # 获取最大的topNfeat维特征
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    # 获取特征值对应的由大到小的topNfeat个特征向量
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    # 将数据转换到低维新空间
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    # 重构数据，用于调试
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


# 使用平均值替换NaN空值
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat
