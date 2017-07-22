# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/22
import unittest

from numpy import *

import recommend
import sim
import svd_rec


class test(unittest.TestCase):
    def test_recom(self):
        myMat = mat(svd_rec.loadExData())
        print ("\n myMat == %s" % (myMat))
        myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
        myMat[3, 3] = 2
        print ("\n myMat == %s" % (myMat))
        recommend_mat = recommend.recommend(myMat, 2)
        # [(2, 2.5), (1, 2.0243290220056256)]
        # 表面用户2对物品2的评分为2.5分，对物品1的预测评分值为2.024
        print ("\n recommend_mat == %s" % (recommend_mat))

        recommend_mat = recommend.recommend(myMat, 2, simMeas=sim.ecludSim)
        print ("\n recommend_mat == %s" % (recommend_mat))

        recommend_mat = recommend.recommend(myMat, 2, simMeas=sim.pearsSim)
        print ("\n recommend_mat == %s" % (recommend_mat))

    def test_svd_recommand(self):
        myMat = mat(svd_rec.loadExData2())
        u, sigma, vT = linalg.svd(myMat)
        print ("\n sigma == %s" % (sigma))

        # 计算多少个奇异值才能超过90%能量
        # 获取平方
        sig2 = sigma ** 2
        sum2 = sum(sig2)
        # sum的0.9
        sum290 = sum2 * 0.9
        print ("\n sum290 == %s" % (sum290))
        # 计算前两个及前三个能量总和
        sig_1 = sum(sig2[:2])
        sig_2 = sum(sig2[:3])
        print ("\n sig_1 == %s\n\n" % (sig_1))
        print ("\n sig_2 == %s\n\n" % (sig_2))
