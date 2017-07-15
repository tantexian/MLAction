# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/14
from numpy import *
import matplotlib.pyplot as plt


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
                # 随机选择第二个alpha值j（不等于i）
                j = selectJrand(i, m)
                # 就按第二个alpha值j对应的预测值
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                # 获取误差
                Ej = fXj - float(labelMat[j])
                # 保存当前i，j的值
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 下述判断保证alpha[j]在0，C之间
                if (labelMat[i] != labelMat[j]):  # 当y1,y2不相等，在分别在分类平面两侧
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print "L==H"; continue
                # eta我alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0: print "eta>=0"; continue
                # 计算获取新的alpha[j]
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                # 调整alpha aj的值，使值在(H,L)区间
                alphas[j] = clipAlpha(alphas[j], H, L)
                # 判断alpha[j]是否有轻微的变化了，没有则继续循环新一轮获取alpha[j]
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                # 如果alphas[j]有了变化
                # 设置使得alphas[]与alphas[j]的值相同，但方向相反（满足一个增大，另一个减少）
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                # the update is in the oppostie direction
                # 计算获取常数项b
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
                # 设置为已更新
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
        # 如果alphas对没有更新，则已迭代次数++
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            # 如果alpha对值已经更新，则设置已迭代次数为零，继续下一轮迭代
            iter = 0
        print "iteration number: %d" % iter
    # 只有在所有数据集上遍历maxIter次，且alpha值不再发生变化 ，程序才会停止并退出while循环
    return b, alphas


# 根据alpha求解w
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# 根据alpha，w 及样本数据绘图
def plot_smo(dataMat, labelMat, ws, b, alphas):
    dataMat = mat(dataMat)
    b = array(b)[0]  # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n = shape(dataMat)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        print("\n labelMat[i] == %s" % (labelMat[i]))
        if int(labelMat[i]) == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])

    # 二维x，y轴画图，xcord1表示个样本数据的x坐标点构成的向量，ycord1为y坐标点构成的向量
    ax.scatter(xcord1, ycord1, s=30, c='orange', marker='s')

    # label == 0 的样本数据画图
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0])  # 注意flatten的用法
    # ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0])  # 注意flatten的用法
    x = arange(-1.0, 10.0, 0.1)  # x最大值，最小值根据原数据集dataArr[:,0]的大小而定
    y = (-b - ws[0][0] * x) / ws[1][0]  # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0,x2就是y值
    ax.plot(x, y)
    for i in range(100):  # 找到支持向量，并在图中标红
        if alphas[i] > 0.0:
            ax.plot(dataMat[i, 0], dataMat[i, 1], 'ro')
    plt.show()


