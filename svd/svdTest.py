# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/21
import unittest

from numpy import *

import svd_rec


class test(unittest.TestCase):
    def test_simple_svd(self):
        data = [[1, 1], [7, 7], [2, 2]]
        # data(m*n) = u(m*m) * sigma(m*n) * vT(n*n)
        # data(3*2) = u(3*3) * sigma(3*2) * vT(2*2)
        u, sigma, vT = linalg.svd(data)
        print ("\n u == %s" % (u))
        # sigma == [ 10.   0.],正常sigma应该返回矩阵sigma == （[ 10.   0.],[ 0.   0.]）
        # numpy 为了节省空间将0元素省略了
        print ("\n sigma == %s" % (sigma))
        print ("\n vT == %s" % (vT))

    def test_svd(self):
        data = svd_rec.loadExData0()
        print ((shape(data)))  # 7*5
        # data(m*n) = u(m*m) * sigma(m*n) * vT(n*n)
        u, sigma, vT = linalg.svd(data)
        # 观察结果simga
        # sigma == [  9.72140007e+00   5.29397912e+00   6.84226362e-01   1.70188300e-16 5.01684085e-47]
        #  其中前面两位数的数量级相对于后面三位大了很多。
        # 因此说明数据集中仅有3个重要特征，其他则都是噪声，或者冗余特征。
        # data(m*n) = u(m*m) * sigma(m*n) * vT(n*n)
        # data(7*5) ~= u(7*3) * sigma(3*3) * vT(3*3)
        print ("\n u == %s" % (u))
        print ("\n vT == %s" % (vT))

        print ("\n sigma == %s" % (sigma))

        sig3 = mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[2]]])
        # 因为numpy为了节省空间将0元素省略了，因此还原sigma矩阵格式sig3
        print ("\n sig3 == %s" % (sig3))

        t = u[:, :3] * sig3 * vT[:3, :]
        print ("\n t == %s" % (t))
