# coding=utf-8
from numpy import tile

import kNN
import unittest


class TestkNN(unittest.TestCase):
    def test_create(self):
        group, labels = kNN.createDateSet()
        print("group == %s\n labels == %s" % (group, labels))
        return

    def test_classify0(self):
        group, labels = kNN.createDateSet()
        # 根据输入的【0，0】判别出属于哪种类型（labels）
        result = kNN.classifyO([0, 0], group, labels, 3)
        print("result == %s" % (result))

    def test_tile(self):
        intX = [1,2]
        x = tile(intX, 2)
        print("\ntile(intX) == %s" %(x))

if __name__ == '__main__':
    unittest.main()
