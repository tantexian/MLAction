# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/15
from numpy import *
from knn import kNN
import optsmo


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = kNN.img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = optsmo.smoOpt(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = optsmo.kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount) / m)
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = optsmo.kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount) / m)
