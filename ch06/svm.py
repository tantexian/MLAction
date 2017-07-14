# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/14
from numpy import *


# 加载构造样本数据矩阵
def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    # 解析文本数据到数据矩阵及标签矩阵中
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


# 功能：随机选择第二个alpha（不等于i）
# i : 第一个alpha的下标
# m : 所有alpha的数目
# 返回：
# j : 第二个alpha值
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/14
def selectJrand(i, m):
    # 希望随机选择的J与i不相等
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 调整alpha aj的值，使值在(H,L)区间
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 功能：简化版SVM(SMO)算法
# 输入：
# dataMatIn : 样本数据集
# classLabels : 类别标签
# C : 控制参数（惩罚参数）
# toler : 容错率
# maxIter : 最大循环次数
# 返回：
# b : f(x)中的b
# alpha : 拉格朗日乘子
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/14
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 将array数组，转换为矩阵
    dataMatrix = mat(dataMatIn);
    # 获取label转置后的列向量
    labelMat = mat(classLabels).transpose()
    b = 0;
    # 获取样本数据的行，列
    m, n = shape(dataMatrix)
    # 初始化alpha为m*1的行矩阵
    alphas = mat(zeros((m, 1)))
    # 初始化迭代次数
    iter = 0
    # 迭代maxIter次
    while (iter < maxIter):
        # 记录alpha是否巳经进行优化
        alphaPairsChanged = 0
        for i in range(m):
            # multiply为矩阵对应元素相乘（*才为矩阵相乘）
            # dataMatrix[i, :] 表示获取矩阵第行，所有列数据
            # 预测当前的样本数据类别为fXi
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算预测值与真实值差值，得到误差Ei
            Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions
            # 一旦alphas等于0或C，那么它们就巳经在“边界”上了，因而不再能够减小或增大，因此也就不值得再对它们进行优化了
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择第二个alpha（不等于i）
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    return b, alphas
