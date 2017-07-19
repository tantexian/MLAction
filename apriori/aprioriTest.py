# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/19
import unittest

import aprioridemo


class test(unittest.TestCase):
    def test_frozenset(self):
        C1 = [[1], [2], [3], [6], [5], [4], [3], [1]]
        C1.sort()
        l = map(frozenset, C1)
        print ("\n l == %s" % (l))

    def test_scand(self):
        data_set = aprioridemo.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        c1 = aprioridemo.createC1(data_set)
        print ("\n map == %s" % (c1))

        D = map(set, data_set)
        print ("\n D == %s" % (D))

        l1, suppData0 = aprioridemo.scanD(D, c1, 0.5)
        print ("\n l1 == %s" % (l1))
        print ("\n suppData0 == %s" % (suppData0))

    def test_apriori(self):
        data_set = aprioridemo.loadDataSet()
        print ("\n data_set == %s" % (data_set))
        l, suppData0 = aprioridemo.apriori(data_set)
        print ("\n l == %s" % (l))

        for i in range(len(l)):
            print ("\n l" + str(i) + " == %s" % (l[i]))

        print ("\n suppData0 == %s" % (suppData0))

        l, suppData0 = aprioridemo.apriori(data_set, minSupport=0.7)
        print ("\n l == %s" % (l))

        for i in range(len(l)):
            print ("\n l" + str(i) + " == %s" % (l[i]))

        print ("\n suppData0 == %s" % (suppData0))
