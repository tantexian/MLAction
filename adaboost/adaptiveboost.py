# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/16

from numpy import *


# 构建样本数据矩阵及类别标签矩阵
def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# 功能：通过阈值比较对数据进行分类，分类为-1或者1
# stump为树桩，即代表单层决策树
# 输入：
# dataMatrix : 样本矩阵数据
# dimen : 矩阵对应的列（即某列特征值）
# threshVal : 用来分类的阈值
# threshIneq : 用于标记小于还是大于（可取值：lt或者gt）
# 返回：
# retArray : 如果参数为(lt)判断小于等于，那么特征值如果小于等于阈值，对应位置设置为-1.0
#            如果参数为(gt) 判断大于，那么特征值如果大于阈值，则对应位置也应该设置为 - 1.0
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/16
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    # 初始化设置为1
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # dataMatrix[:, dimen] <= threshVal：将列向量中每个元素与threshVal比较（是否<=），
        # 如果小于等于则对应值设置为True，否则设置为False
        # 获取上述值之和，对于所有为True的元素值对应retArry相应位置的值设置为-1.0
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 遍历stumpClassify所有可能的输入值，找出数据集上最佳的单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    # 获取样本数据的行和列
    m, n = shape(dataMatrix)
    # 用于在特征值的所有可能特征值进行遍历
    numSteps = 10.0
    # 用于存储给定权向量D时所能得到的最佳单层决策树的相关信息
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    # 初始化为无穷大，用于后续寻找可能的最小错误率
    minError = inf  # init error sum, to +infinity
    # 在程序所有特征值集进行遍历
    for i in range(n):  # loop over all dimensions
        # 获取第i个特征中的最小值及最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 由于样本数据是数值型，通过计算最大值和最小值来了解应该需要多大的步长
        stepSize = (rangeMax - rangeMin) / numSteps
        # 间隔迭代计算stepSize次
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            # 在大于和小于之间切换不等式
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                # 依次计算得出间隔数值
                threshVal = (rangeMin + float(j) * stepSize)
                # 通过阈值threshVal比较对数据进行分类，返回分类预测结果
                # 判断第i列特征集，对应位的值是否满足（lt，gt）小于等于或者大于阈值，
                # 如果上述条件满足，则predictedVals数组对应位置设置为-1.0
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                errArr = mat(ones((m, 1)))
                # 将真实label值与预测值相等(为True)的位置置0
                errArr[predictedVals == labelMat] = 0
                # 计算加权错误率，TODO:  此处公式相乘更新权重D的解释？
                # 如果使用其他分类器的话，就需要考虑D上最佳分类器所有定义的计算过程
                # D.T为1*m的行向量， errArr为m*1的列向量，weightedError为单个值[[val]]
                weightedError = D.T * errArr  # calc total error multiplied by D
                print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                    i, threshVal, inequal, weightedError)
                # 如果当期weightedError<minError,更新minError
                if weightedError < minError:
                    minError = weightedError
                    # 全新拷贝分类预测结果
                    bestClasEst = predictedVals.copy()
                    # 保存在给定权向量D时每次循环获取到的最佳单层决策树的相关信息
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


# 功能：xx
# 输入：
# dataArr : 数据集
# classLabels : 类标签
# numIt : 迭代次数（如果误差到0则提前结束），整个算法唯一需要用户指定的参数
# 返回：
# xx : xx
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/16
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    # 获取行
    m = shape(dataArr)[0]
    # 初始化为相同权重
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    # 迭代numIt次
    for i in range(numIt):
        # 获取数据集上最佳的单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        print "D:",D.T
        # 计算步长alpha
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        # 保存当前alpha值
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        print "classEst: ",classEst.T
        # 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        # 错误率累加计算
        aggClassEst += alpha * classEst
        print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0: break
    return weakClassArr
