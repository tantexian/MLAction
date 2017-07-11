# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/11
import unittest

import matplotlib
import matplotlib.pyplot as plt


class TestMatplotlib(unittest.TestCase):
    def test_plot(self):
        fig = plot.figure()
        ax = fig.add_subplot(111)
