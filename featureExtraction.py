#encoding=utf8
__author__ = 'ZGD'

import numpy as np
from sklearn.feature_selection import SelectKBest,chi2

def chi2Test(tfIdfMat,labels):
    """

    :param tfIdfMat:
    :param labels:
    :return:
    """
    #根据chi2方法选择6000个最好的特征
    ch2 = SelectKBest(chi2,k=6000)
    trainSet = []
    for i in range(len(tfIdfMat[0])):
        trainSet.append(tfIdfMat[:,i])
    ch2_sx_np  = ch2.fit_transform(trainSet,labels)
    # ch2_sx_np = np.array(ch2_sx_np)
    # labels = np.array(labels)
    return ch2_sx_np,labels


def ChiCalc(a, b, c, d):
    """
    卡方计算公式
    # a：在这个分类下包含这个词的文档数量
    # b：不在该分类下包含这个词的文档数量
    # c：在这个分类下不包含这个词的文档数量
    # d：不在该分类下，且不包含这个词的文档数量
    :param a:
    :param b:
    :param c:
    :param d:
    :return:
    """
    result = float(pow((a*d - b*c), 2)) /float((a+c) * (a+b) * (b+d) * (c+d))
    return result


def calculateChi(bowMat,labels,vocab):

    """
    计算多个类别的卡方
    :param bowMat:行是词,列是文档,表示词的数量
    :param labels:10个类,计算
    :return:
    """

    labelDict = {'acq':0,'corn':1,'crude':2,'earn':3,'grain':4,'interest':5,'money-fx':6,'ship':7,'trade':9,'wheat':10}
    wordChiDict = {}
    ll = len(labels)
    labelType = set(labels)
    totallen = len(bowMat)
    for w in vocab:

        iindex = vocab.index(w)
        number = 0
        chiSum = 0.0
        for type in labelType:
            a = 0
            b = 0
            c = 0
            d = 0
            for j in range(len(bowMat[0])):
                # index = labelDict[labels[j]]

                if bowMat[iindex][j]!=0:

                    if labels[j] == type:
                        a = a+1
                        number +=1
                    else:
                        b = b+1

                else:
                    if labels[j] == type:
                        c = c+1
                        number +=1
                    else:
                        d = d+1

            chiSum=chiSum+number*ChiCalc(a, b, c, d)/totallen
        wordChiDict[w]=chiSum
    return wordChiDict

def selectBestKByChi(vocab,wordChiDict,k=7000):
    """

    :param vocab:
    :param wordChiDict:
    :param k:
    :return:
    """
    featureVocab = []
    sortedWordDict = sorted(wordChiDict.items(),key=lambda item:item[1],reverse=True)
    if len(sortedWordDict)<k:
        return wordChiDict.keys()
    else:
        for i in range(k):
            featureVocab.append(sortedWordDict[i][0])
    return featureVocab

def selectBestKByEntropy(wordEntDict,k=7000):
    """

    :param vocab:
    :param wordChiDict:
    :param k:
    :return:
    """
    featureVocab = []
    sortedWordDict = sorted(wordEntDict.items(), key=lambda item: item[1], reverse=True)
    if len(sortedWordDict) < k:
        return wordEntDict.keys()
    else:
        for i in range(k):
            featureVocab.append(sortedWordDict[i][0])
    return featureVocab


def calEntropy(y):
    '''
    功能：calEntropy用于计算香农熵 e=-sum(pi*log pi)
    参数：其中y为数组array
    输出：信息熵entropy
    '''
    n = len(y)
    labelCounts = {}
    for label in y:
        if label not in labelCounts.keys():
            labelCounts[label] = 1
        else:
            labelCounts[label] += 1
    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / n
        entropy -= prob * np.log2(prob)
    return entropy

def calculateEntropy(bowMat,labels,vocab):
    """

    :param bowMat:
    :param labels:
    :param vocab:
    :return:
    """
    labelDict = {'acq': 0, 'corn': 1, 'crude': 2, 'earn': 3, 'grain': 4, 'interest': 5, 'money-fx': 6, 'ship': 7,
                 'trade': 8, 'wheat': 9}
    wordEntDict = {}

    entropy = calEntropy(labels)
    for w in vocab:

        iindex = vocab.index(w)

        a = np.zeros(10)
        for j in range(len(bowMat[0])):
            #类型
            index = labelDict[labels[j]]
            if bowMat[iindex][j] != 0:

                a[index] += 1

        IG = entropy - Entrocal(a)
        wordEntDict[w] = IG
    return wordEntDict

def Entrocal(a):

    """

    :param a:
    :return:
    """
    Entropysum = 0.0
    totalNum = np.sum(a)


    for n in a :
        Entropysum += np.log2((1+n*1.0)/totalNum)*n*1.0/totalNum
    return Entropysum