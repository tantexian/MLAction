# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/20
import unittest

import matplotlib.pyplot as plt
from numpy import *

import pcademo


class test(unittest.TestCase):
    def test_pca(self):
        data_set = pcademo.loadDataSet("testSet.txt")
        m, n = shape(data_set)
        print ("\n shape(data_set) == %s * %s" % (m, n))
        print ("\n data_set == %s" % (data_set))

        # 如果此处参数为2，则返回2维特征
        # （而数据本身就只有2维特征，因此所有特征全部返回，所以所有点全部为红色与原始数据重合）
        lowDDataMat, reconMat = pcademo.pca(data_set, 1)
        m, n = shape(reconMat)
        print ("\n shape(reconMat) == %s * %s" % (m, n))
        print ("\n reconMat == %s" % (reconMat))

        m, n = shape(lowDDataMat)
        print ("\n shape(lowDDataMat) == %s * %s" % (m, n))
        print ("\n lowDDataMat == %s" % (lowDDataMat))

        figure = plt.figure()
        ax = figure.add_subplot(111)
        # flatten()折叠成一维的对象
        ax.scatter(data_set[:, 0].flatten().A[0], data_set[:, 1].flatten().A[0], marker="^", s=90)

        ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50, c="red")

        plt.show()

    # 在 eig_vals 结果集中我们发现了很多0值，意味着这些特征都是其他特征的副本，
    # 也就是说它们可以通过其他特征表示，而自身没有提供更多额外的信息。
    # 最前面15个值的数量级大于105，实际上那以 后的值都变得非常小。
    # 这就相当于告诉我们只有部分重要特征，重要特征的数目也很快就会下降。
    # 最后, 我们可能会注意到有一些小的负值，它们主要源自数值误差应该四舍五入成0。
    def test_semiconductor(self):
        data_set = pcademo.replaceNanWithMean()
        # print ("\n data_set == %s" % (data_set))

        meanVal = mean(data_set, axis=0)
        mean_removed = data_set - meanVal
        # 计算协方差
        cov_mat = cov(mean_removed, rowvar=0)
        eig_vals, eig_vects = linalg.eig(cov_mat)
        print ("\n eig_vals ==\n %s" % (eig_vals))

    def test_mean(self):
        data_mat = arange(6).reshape(2, 3)
        print ("\n data_mat == %s" % (data_mat))

        data_mean = mean(data_mat, axis=0)
        print ("\n data_mean == %s" % (data_mean))

        sub = data_mat - data_mean
        print ("\n sub == %s" % (sub))

    def test_cov(self):
        data_mat = [[1, 2, 3],
                    [0, 1, 2],
                    [3, 2, 0]]
        print ("\n data_mat == %s" % (data_mat))

        data_mean = mean(data_mat)
        print ("\n data_mean == %s" % (data_mean))

        # 求方差
        varVal = var(data_mat)
        print ("\n varVal == %s" % (varVal))

        # 求协方差 Fixme: ??? cov如何求解？
        # 求协方差矩阵（更多请参考博文：https://my.oschina.net/tantexian/blog/1476444）
        covVal = cov(data_mat)
        print ("\n covVal ==\n %s" % (covVal))

    def test_eig(self):
        data_mat = [[1, 2, 3],
                    [0, 1, 2],
                    [3, 2, 0]]
        print ("\n data_mat == %s" % (data_mat))

        eigs = linalg.eig(mat(data_mat))
        print (eigs)
