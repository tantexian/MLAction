# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/13
import unittest

import logRegres


class colicTest(unittest.TestCase):
    def test_colic_classfiy(self):
        logRegres.colicTest()

    def test_colic_mult(self):
        logRegres.multiTest()
