# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
from numpy import *


# 创建实验样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 代表侮辱性文字， 0 代表正常言论
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


# 获取文档中出现的所有词的列表（不重复）
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 将每篇文章中返回的新词添加到列表集合中
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


# 返回文档向量（0、1分别表示单词是否在文档中出现）
# vocabList 词汇表
# inputSet  需要变成向量化的某个文档
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个词汇表等长的向量，初始化为0
    returnVec = [0] * len(vocabList)
    # 遍历文档中所有单词
    for word in inputSet:
        # 单词出现在词汇表中，设置该位置向量值为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


# 返回文档向量（数字分别表示单词是在文档中出现次数）
# vocabList 词汇表
# inputSet  需要变成向量化的某个文档
def bagOfWords2Vec(vocabList, inputSet):
    # 创建一个词汇表等长的向量，初始化为0
    returnVec = [0] * len(vocabList)
    # 遍历文档中所有单词
    for word in inputSet:
        # 单词出现在词汇表中，设置该位置向量值为1
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec


# trainMatrix 训练文档矩阵
# trainCategory 每篇文档类别标签，构成的向量（类似：classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not）
def trainNB0(trainMatrix, trainCategory):
    # 总文档数量
    numTrainDocs = len(trainMatrix)
    # 文档单词数
    numWords = len(trainMatrix[0])
    # 利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率 ，
    # 即计算p(w0|1)p(w1|1)p(w2|1)。如果其中一个概率值为0,那么最后的乘积也为0。
    # 为降低 这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
    # 向量中1表示出现侮辱性单词，则sum(trainCategory)表示出现侮辱词语总数
    # pAbusive 侮辱性词语出现概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化概率，构建全1矩阵，表示对应单词向量出现的次数
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    # p0Denom，p1Denom表示对应侮辱性和正常词语类别词语总数，设置初值为2.0
    p0Denom = 2.0
    p1Denom = 2.0  # change to 2.0
    # 遍历文档向量
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 如果为1，则表示侮辱性词语，对应计数增加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            # 如果为0，则表示正常言论，对应计数增加
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 正常应该是直接相除：p1Vect = p1Num / p1Denom
    # 由于概率太小比较小的数相乘，程序可能会下溢。因此采用log对数形式来避免
    p1Vect = log(p1Num / p1Denom)  # change to log()
    p0Vect = log(p0Num / p0Denom)  # change to log()
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类，根据输入的数量向量，输出属于哪个类别
# vec2Classify 测试数据向量
# p0Vec p0概率向量数组
# p1Vec p1概率向量数组
# pClass1 实验整体样本数据，为class1分类的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 计算当前数据为p0,p1各自的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    # 哪个概率大返回哪个
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# 测试朴素贝叶斯分类
def testingNB():
    # 获取实验样本数据
    listOPosts, listClasses = loadDataSet()
    # 获取文档中出现的所有词的列表（不重复）
    myVocabList = createVocabList(listOPosts)
    # 获取训练数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 测试数据集
    testEntry = ['love', 'my', 'dalmation']
    # 获取测试数据集的向量列表
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    # classifyNB 为判断测试数据，属于哪个分类
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)
