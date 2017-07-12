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

    def test_one_zeros(self):
        a = ones(10)
        print ("\n a == %s" % (a))
        a = zeros(10)
        print ("\n a == %s" % (a))
