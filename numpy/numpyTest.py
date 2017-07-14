# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/11
import unittest

from numpy import *


class numpyTest(unittest.TestCase):
    def test_tile(self):
        intX = [1, 2]
        # 行方向重复2次，列方向重复3次
        x = tile(intX, (2, 3))
        print("\ntile(intX) == %s" % (x))

    def test_sum(self):
        a = array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        print "\nnumpy.sum == \n"
        print a.sum(axis=0)
        print a.sum(axis=1)
        # print a.sum(axis=2)

    def test_sum1(self):
        print "\na == \n"
        a = arange(24).reshape(4, 6)
        print a

        # 按照各列相加取和
        print "\na.sum(axis = 0) == \n"
        x = a.sum(axis=0)
        print x

        # 按照各行相加取和
        print "\na.sum(axis = 1) == \n"
        x = a.sum(axis=1)
        print x

    def test_zero(self):
        # 构建一个3*4的零矩阵
        a = zeros((3, 4))
        print("\n a == %s" % (a))

    def test_index(self):
        a = arange(10)
        # 打印数组最后一个值
        print a[-1]

    # shape函数是numpy.core.fromnumeric中的函数，它的功能是读取矩阵的长度，
    # 比如shape[0]就是读取矩阵第一维度的长度。它的输入参数可以使一个整数表示维度，也可以是一个矩阵。
    # 如果是数组，则返回（函数，），列数不存在
    # 如果是矩阵，则返回矩阵的（行数，列数）
    def test_shape(self):
        a = arange(10)
        print ("\n a == %s" % (a))
        print ("\n a.shape == %s a.shape[0] == %s\n" % (a.shape, a.shape[0]))

        a = arange(24).reshape(4, 6)
        print ("\n a == %s" % (a))
        print ("\n a.shape == %s a.shape[0] == %s\n" % (a.shape, a.shape[0]))

        a = arange(1).reshape(1, 1)
        print ("\n a == %s" % (a))
        print ("\n a.shape == %s a.shape[0] == %s\n" % (a.shape, a.shape[0]))

    def test_ones(self):
        # error
        # a = ones(3,4);
        # print ("\n a == %s" % (a))
        a = ones((3, 4));
        print ("\n a == %s" % (a))

    # python的[[]]乘法和矩阵乘法不一致
    def test_mult(self):
        a = [[1, 2],
             [3, 4]]
        b = [[1],
             [2]]
        one = ones((2, 2))
        mult = a * one
        print ("\n mult == %s" % (mult))

        mult = mat(a) * mat(one)
        print ("\n mult == %s" % (mult))

        # error?
        mult = mat(a) * mat(b)
        print ("\n mult == %s" % (mult))

    def test_mul(self):
        # a==3*2 b==2*1 则a*b==3*1
        a = [[1, 2],
             [3, 4],
             [5, 6]]

        b = [[1],
             [2]]

        c = mat(a) * mat(b)

        print ("\n c == %s" % (c))

    # 转秩矩阵（即沿着对角线交换矩阵对称位置数据）
    def test_transpose(self):
        a = arange(4).reshape(2, 2)
        a = mat(a)
        print ("\n a == %s" % (a))
        print ("\n a.transpose() == %s" % (a.transpose()))
        print ("\n a.T == %s" % (a.T))

    def test_multiply(self):
        a = arange(6).reshape(3, 2)
        a = mat(a)
        print ("\n a == %s" % (a))

        b = arange(6).reshape(3, 2)
        b = mat(b)
        print ("\n b == %s" % (b))

        # multiply为对应元素相乘
        t = multiply(a, b)
        print ("\n t == %s" % (t))

        c = arange(4).reshape(2, 2)
        c = mat(c)
        print ("\n b == %s" % (c))

        # * 为矩阵相乘
        t2 = a * c
        print ("\n t2 == %s" % (t2))

    def test_split(self):
        a = arange(15).reshape(5, 3)
        a = mat(a)
        print ("\n a == %s" % (a))
        # 输出第3行（从0开始），所有列
        print ("\n a[3, :] == %s" % (a[3, :]))
