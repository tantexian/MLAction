import kNN
import unittest


class TestkNN(unittest.TestCase):
    def test_create(self):
        group, labels = kNN.createDateSet()
        print("group == %s\n labels == %s" % (group, labels))
        return

    def test_classify0(self):
        group, labels = kNN.createDateSet()
        result = kNN.classifyO([0, 0], group, labels, 3)
        print("result == %s" % (result))


if __name__ == '__main__':
    unittest.main()
