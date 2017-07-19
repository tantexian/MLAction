# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/19

from numpy import *


# 加载样本数据
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


# 根据样本数据的每列特征构建一个包含k个随机质心的集合
# 返回k行n列数组(其中每列特征值数据为随机生成范围在原始数据特征值的min~max之间)
def randCent(dataSet, k):
    # 获取数据列
    n = shape(dataSet)[1]
    # 构建全零矩阵（k行n列）
    centroids = mat(zeros((k, n)))  # create centroid mat
    # 遍历n列数据
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        # 获取当前第j列特征值中最小的值
        minJ = min(dataSet[:, j])
        # 寻找当前第j列特征值的范围（最大最小值的间隔）
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # random.rand(k, 1)随机生成k*1的矩阵（每个值在0~1中随机生成）
        # 随机生成的特征列j保证在最小值~最大值的范围之内
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


# 功能：根据簇数目k，返回所有质心类及分配结果
# dataSet : 样本数据
# k : 簇数据
# distMeas : 距离计算函数（默认为欧式距离）
# createCent : 根据样本数据的每列特征构建一个包含k个随机质心的集合函数
# 返回：
# centroids : 返回k行的随机质心集合
# clusterAssment : 簇分配结果矩阵（第一列存储簇索引，第二列存储误差（当前点到簇质心的距离））
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/19
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取样本数据行数
    m = shape(dataSet)[0]
    # m*2的全零矩阵（用于存储每个点的簇分配结果）
    # clusterAssment簇结果矩阵包含两列：第一列存储簇索引，第二列存储误差（当前点到簇质心的距离）
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    # 根据样本数据的每列特征构建一个包含k个随机质心的集合
    centroids = createCent(dataSet, k)
    clusterChanged = True
    # clusterChanged = True表示需要计算距离
    while clusterChanged:
        # 以及进入计算的值，不需要再次计算
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf;
            minIndex = -1
            for j in range(k):
                # 迭代计算随机质心集合中的第j行的向量与样本数据第i行的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 选取出随机质心各行中离所有样本欧式距离最近的行minIndex
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                # 任意簇的分配结果发生改变，更新clusterChanged。表示需要重新迭代计算
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        # 遍历所有质心并更新它们的值
        for cent in range(k):  # recalculate centroids
            # 通过数组过滤来获取给定簇cent的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            # 计算所有点的均值（axis=0表示沿矩阵列方向进行均值计算）
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment
