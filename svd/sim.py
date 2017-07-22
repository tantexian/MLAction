# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/21
from numpy import *


# 欧氏距离相关度计算
def ecludSim(inA, inB):
    return 1.0 / (1.0 + linalg.norm(inA - inB))


# 皮尔逊相关度计算
def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦相关度计算
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = linalg.norm(inA) * linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)
