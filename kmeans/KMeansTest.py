# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/19

import unittest

import matplotlib.pyplot as plt
from numpy import *

import KMeans


class MyTestCase(unittest.TestCase):
    def test_rand_cent1(self):
        data_mat = mat(KMeans.loadDataSet("testSet.txt"))
        min0 = min(data_mat[:, 0])
        max0 = max(data_mat[:, 0])
        print ("\n min0 == %s max0 == %s" % (min0, max0))
        min1 = min(data_mat[:, 1])
        max1 = max(data_mat[:, 1])
        print ("\n min1 == %s max1 == %s" % (min1, max1))

        cent = KMeans.randCent(data_mat, 2)
        print ("\n cent == %s" % (cent))

        dist_eclud = KMeans.distEclud(data_mat[0], data_mat[1])
        print ("\n dist_eclud == %s" % (dist_eclud))

    def test_kmeans(self):
        data_mat = mat(KMeans.loadDataSet("testSet.txt"))
        centroids, clusterAssment = KMeans.kMeans(data_mat, 4)
        print ("\n centroids == \n %s" % (centroids))
        print ("\n clusterAssment == \n %s" % (clusterAssment))

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xcord = data_mat[:, 0].A
        ycord = data_mat[:, 1].A

        xCluster0 = []
        yCluster0 = []
        xCluster1 = []
        yCluster1 = []
        xCluster2 = []
        yCluster2 = []
        xCluster3 = []
        yCluster3 = []

        for i in range(shape(clusterAssment)[0]):
            if clusterAssment[i, 0] == 0:
                xCluster0.append(data_mat[i, 0])
                yCluster0.append(data_mat[i, 1])
            elif clusterAssment[i, 0] == 1:
                xCluster1.append(data_mat[i, 0])
                yCluster1.append(data_mat[i, 1])
            elif clusterAssment[i, 0] == 2:
                xCluster2.append(data_mat[i, 0])
                yCluster2.append(data_mat[i, 1])
            elif clusterAssment[i, 0] == 3:
                xCluster3.append(data_mat[i, 0])
                yCluster3.append(data_mat[i, 1])
        ax.scatter(xCluster0, yCluster0, s=30, c='orange', marker='s')
        ax.scatter(xCluster1, yCluster1, s=30, c='red', marker='p')
        ax.scatter(xCluster2, yCluster2, s=30, c='blue', marker='*')
        ax.scatter(xCluster3, yCluster3, s=30, c='black', marker='d')

        # 绘制原始数据
        # ax.scatter(xcord, ycord, s=10, c='orange', marker='s')

        # 绘制质心点
        xcord2 = centroids[:, 0].A
        ycord2 = centroids[:, 1].A
        ax.scatter(xcord2, ycord2, s=100, c='red', marker='+')
        plt.show()

    def test_random(self):
        rand = random.rand(2, 3)
        print ("\n rand == %s" % (rand))

    def test_rand_cent(self):
        dataMat = arange(24).reshape(3, 8)
        print ("\n dataMat == \n %s" % (dataMat))
        centroids = KMeans.randCent(dataMat, 8)
        print ("\n centroids == \n %s" % (centroids))
