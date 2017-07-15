# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/12
import trees
import treePlotter


def eyesTree():
    fr = open("lenses.txt")
    # 按照tab分割数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic1', 'tearRate']
    lensesTree = trees.createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)
