# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/21
import unittest

from numpy import *

import sim
import svd_rec


class test(unittest.TestCase):
    def test_sim(self):
        data = mat(svd_rec.loadExData0())
        intA = data[:, 0]
        intB = data[:, 4]
        eclud_sim = sim.ecludSim(intA, intB)
        print ("\n eclud_sim == %s" % (eclud_sim))

        pears_sim = sim.pearsSim(intA, intB)
        print ("\n pears_sim == %s" % (pears_sim))

        cos_sim = sim.cosSim(intA, intB)
        print ("\n cos_sim == %s" % (cos_sim))
