# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/21
from numpy import *

import sim


# 功能：给定相似度计算方法的条件下，计算用户对物品估计评分值
# dataMat : 样本数据矩阵
# user : 用户编号
# item : 物品编号
# simMeas : 计算相似度方法
# 返回值：
# xxx : xxx
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/21
def standEst(dataMat, user, simMeas, item):
    # 样本矩阵行对应用户，列对应物品
    # 获取物品数
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    # 遍历行中每个物品
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        # 寻找两个用户都评价的物品
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                 dataMat[overLap, j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 功能：用于代替standEst，对给定用户物品构建一个评分估计值
#      与standEs不同在于使用了SVD分解，只包含90%能量的奇异值，
#      利用这些奇异值构建对角矩阵，然后利用u矩阵将物品转换到低维空间
# dataMat : 样本数据矩阵
# user : 用户编号
# item : 物品编号
# simMeas : 计算相似度方法
# 返回值：
# xxx : xxx
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/22
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    # 建立对角矩阵
    U, Sigma, VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix
    # 构建转换后的物品
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items
    # 在用户对应行进行遍历
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=sim.cosSim, estMethod=standEst):
    # 寻找未评级的物品
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 寻找前N个未评级物品
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# 打印矩阵
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print 1,
            else:
                print 0,
        print ''


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)
    U, Sigma, VT = linalg.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print "****reconstructed matrix using %d singular values******" % numSV
    printMat(reconMat, thresh)
