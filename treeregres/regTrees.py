# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18

from numpy import *


# 加载样本数据
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行数据映射为浮点数
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


# 功能：如果数据集中的第feature列的特征值大于value则放置到mat0矩阵，否则放到mat1矩阵
# dataSet : 样本数据集
# feature : 代表数据集中样本的第feature列特征值
# value : 分割阈值
# 返回：返回第feature列的特征值根据value阈值分割的两个矩阵
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


# 负责生成叶节点模型（在回归树中，该模型即为目标变量的均值）
def regLeaf(dataSet):  # returns the value used for each leaf
    return mean(dataSet[:, -1])


# 误差估计函数，在给定数据集上面计算数据的总平方误差
def regErr(dataSet):
    # var()为均方差函数（方差*数据个数=总平方误差）
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def linearSolve(dataSet):  # helper function used in two places
    m, n = shape(dataSet)
    X = mat(ones((m, n)));
    Y = mat(ones((m, 1)))  # create a copy of data with 1 in 0th postion
    X[:, 1:n] = dataSet[:, 0:n - 1];
    Y = dataSet[:, -1]  # and strip out Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def modelLeaf(dataSet):  # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 功能：获取数据集上最佳的二元切分方式和生产对应的叶节点
#      遍历所有特征及其可能取值来寻找使误差最小化的切分阈值
# leafType : 建立叶节点的函数的引用
# errType : 总方差计算函数的引用
# ops : 树构建所需其他参数元组，包括两个参数[tolS：允许的误差下降值，tolN：切分的最小样本数]
# 返回：如果寻找到了好的切分方式则返回特征编号和切分特征值
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    # 如果不同剩余特征值所有值相等，则返回退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        # leafTyp函数为计算样本特征值的均值
        return None, leafType(dataSet)
    # 获取样本数据集的行列
    m, n = shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    # errType为计算样本数据的总方差
    S = errType(dataSet)
    # 初始化bestS最优总方差
    bestS = inf
    bestIndex = 0
    bestValue = 0
    # 遍历所有特征列
    for featIndex in range(n - 1):
        # 依次迭代使用featIndex列中的不同特征值
        for splitVal in set(dataSet[:, featIndex]):
            # 使用splitVal阈值来分割样本数据中第featIndex列
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果返回的切分样本数目小于tolN，则继续寻找下一个特征分割阈值
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            # 获取切分后的两个矩阵的总方差和
            newS = errType(mat0) + errType(mat1)
            # 如果总方差小于bestS最优方差，则更新bestS值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    # 如果误差减少不大（不大于允许的误差下降值），则返回退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    # 使用splitVal阈值bestValue来分割样本数据中第bestIndex列
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分的数据集很小，则返回退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split


# 功能：递归函数，尝试将数据集分割成两部分(递归)
# dataSet : 样本数据集
# leafType : 建立叶节点的函数
# errType : 误差计算函数
# ops : 树构建所需其他参数元组
# 返回：xxx
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/18
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):  # assume dataSet is NumPy Mat so we can array filtering
    # 获取最好的切分方式的特征编号和切分特征阈值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # choose the best split
    if feat == None:
        # 满足停止条件时，返回叶节点
        return val  # if the splitting hit a stop condition return val
    retTree = {}
    # 保存最佳分割特征列编号
    retTree['spInd'] = feat
    # 保存最佳分割特征列阈值
    retTree['spVal'] = val
    # 获取样本数据集中第feat列特征值数据根据value阈值分割的两个矩阵
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归构建左子树及右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree



