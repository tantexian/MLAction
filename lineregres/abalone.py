# coding=utf-8
# @author tantexian, <my.oschina.net/tantexian>
# @since 2017/7/17

def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
    return ((yArr - yHatArr) ** 2).sum()
