# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
import unittest

import bayes
import email_nb


class bayesTest(unittest.TestCase):
    def test_createVocablist(self):
        data_set, _ = bayes.loadDataSet()
        vocab_list = bayes.createVocabList(data_set)
        print ("\n vocab_list == %s" % (vocab_list))

        # 根据数据集第0行输出对应的向量表
        # （即，第0行中所有单词，在整个data_set词汇表中出现的单词位置设置为1）
        vec = bayes.setOfWords2Vec(vocab_list, data_set[0])
        print ("\n vec == %s" % (vec))
        vec = bayes.setOfWords2Vec(vocab_list, data_set[3])
        print ("\n vec == %s" % (vec))

    def test_train_nb(self):
        data_set, listClasses = bayes.loadDataSet()
        vocab_list = bayes.createVocabList(data_set)
        print ("\n vocab_list == %s" % (vocab_list))

        trainMat = []
        for postinDoc in data_set:
            trainMat.append(bayes.setOfWords2Vec(vocab_list, postinDoc))

        p0Vect, p1Vect, pAbusive = bayes.trainNB0(trainMat, listClasses)
        print ("\n p0Vect == %s\n p1Vect == %s\n pAbusive == %s\n" % (p0Vect, p1Vect, pAbusive))

    def test_nb(self):
        bayes.testingNB()

    def test_email_nb(self):
        email_nb.spamTest()
