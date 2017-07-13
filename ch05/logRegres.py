# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
from numpy import *


# 返回数据矩阵及label矩阵向量
# 从testSet.txt文本中加载数据
def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # FIXME: 此处为什么在所有数据前面都加上1.0 ???, 将第一个特征都设置为1.0？？？
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid公式计算（y = 1/(1+1/e^x)）
# 注意：此处inX可以是行向量
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 使用梯度上升方式或者最佳回归权重参数
# dataMatIn 2维的numpy数组，每列代表不同的特征，每行代表一个训练数据
# dataMatIn（注意默认加入了一列x0=1 FIXME: Why???）
# classLabels 类标签，1*100 的行向量
def gradAscent(dataMatIn, classLabels):
    # 转换为numpy的矩阵
    dataMatrix = mat(dataMatIn)  # convert to NumPy matrix
    # 将行向量，转换为numpy的列向量
    labelMat = mat(classLabels).transpose()  # convert to NumPy matrix
    # 获取数组的行列
    m, n = shape(dataMatrix)
    # 步长设置为0.001
    alpha = 0.001
    # 最大迭代次数设置为500
    maxCycles = 500
    # 构建n*1全1矩阵
    weights = ones((n, 1))
    # 迭代maxCycles次
    for k in range(maxCycles):  # heavy on matrix operations
        # 使用s函数求值（矩阵相乘:左行乘右列）
        # dataMatrix * weights == wTx == w0x0 + w1x1 + ... + wnxn
        # dataMatrix 代表 xT == x0, x1, ..., xn
        # 第一次weights为全1列向量，因此w0 = w1 = wn == 1(后续每次weight根据步长与x方向及y方向偏移梯度上升)
        h = sigmoid(dataMatrix * weights)  # matrix mult 其中h为列向量，每个值在0~1之间
        # 向量相减（h也为列向量）
        error = (labelMat - h)  # vector subtraction
        # transpose矩阵转秩
        # w = w + alpha * detaw*f(w)
        # 此处公式详细推导请参考博文：https://my.oschina.net/tantexian/blog/1359191
        weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
    return weights


# 使用随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    # 获取数据的行列
    m, n = shape(dataMatrix)
    # 设置步长
    alpha = 0.01
    # 初始权重向量w设置为全1
    weights = ones(n)  # initialize to all ones
    # 遍历所有行
    for i in range(m):
        # 使用s函数求值(与梯度上升不一样的是，这里只是选取其中一行，及一条样本数据（一行中的每个值及该条样本数据的各特征值）)
        # dataMatrix[i] * weights == wTx == w0x0 + w1x1 + ... + wnxn
        # dataMatrix[i] 代表一条样本数据（一行代表一条样本数据） xT == x0, x1, ..., xn
        # 第一次weights为全1列向量，因此w0 = w1 = wn == 1(后续每次weight根据步长与x方向及y方向偏移梯度上升)
        # dataMatrix[i] == 1*n 矩阵  weights == n*1 因此dataMatrix[i] * weights 为 1*1
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 将上述获取的h值与真实分类值比较得出误差
        error = classLabels[i] - h
        # w = w + alpha * detaw*f(w)
        # 此处公式详细推导请参考博文：https://my.oschina.net/tantexian/blog/1359191
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升（每次能够动态改变步长，加快收敛速度）
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    # 迭代numIter次
    for j in range(numIter):
        # dataIndex 为0~m 的数组
        dataIndex = range(m)
        # 迭代遍历m次（m为矩阵行数）
        for i in range(m):
            # 每次动态调整步长参数（随着i,j 越来越大，步长慢慢变小，精度增加）
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            # random.uniform(0,m)获取0~m之间的随机浮点数据
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            # 随机获取矩阵数据中的一行数据（随机获取）
            # 随机选取样本来更新回归参数，将减少周期性波动
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 从列表中删除之前已被使用的值
            del (dataIndex[randIndex])
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    # 获取需要绘制的数据
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    # 获取数据行
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plotBestFit1(weights, dataMat, labelMat):
    import matplotlib.pyplot as plt
    # dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


# logstic归回分类函数
# inX 为样本的特征行向量
# weights 根据已有样本训练获取得到的回归向量
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    # 获取训练数据
    frTrain = open('horseColicTraining.txt');
    # 获取测试数据
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        # 用tab键分割每一行样本数据
        currLine = line.strip().split('\t')
        lineArr = []
        # 依次读取每一行的20个数值
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用随机梯度算法，根据训练样本数据，获取回归参数
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 使用logstic分类回归函数，判断test的每行样本数据，然后和真实结果对比，并计算错误率
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

# 进行多次尝试
def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))
