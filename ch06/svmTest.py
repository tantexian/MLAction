# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/14
import unittest
import svm


class svmTest(unittest.TestCase):
    def test_simple_smo(self):
        data_set, label_set = svm.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = svm.smoSimple(data_set, label_set, 0.6, 0.001, 40)
        print("\n b == %s" % (b))
        print("\n alpha == %s" % (alpha[alpha > 0]))

        for i in range(100):
            if alpha[i] > 0.0:
                print("\n  data_set[i] == %s label_set[i] == %s " % (data_set[i], label_set[i]))

    def test_plot(self):
        data_set, label_set = svm.loadDataSet("testSet.txt")
        print("\n data_set == %s" % (data_set))
        print("\n label_set == %s" % (label_set))
        b, alpha = svm.smoSimple(data_set, label_set, 0.6, 0.001, 40)
        ws = svm.calcWs(alpha, data_set, label_set)
        # 其中红色点为支持向量
        svm.plot_simple_smo(data_set, label_set, ws, b, alpha)
