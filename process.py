#encoding=utf8
__author__ = 'ZGD'
import re
import os
import sys
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy as np


reload(sys)
sys.setdefaultencoding('utf8')

def readCorpus():
    """

    :return:
    """
    starttime = time.clock()
    filePath = u'./training/'
    labels = []
    trainSet = []
    fileList = os.listdir(filePath)
    english_stopwords = stopwords.words('english')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','<','>','-']
    st = LancasterStemmer()
    # print english_stopwords
    numText = 0
    for fileType in fileList:
        filepath_2 = filePath+fileType
        fileList_2 = os.listdir(filepath_2)
        print '处理类别：',fileType
        for fileName in fileList_2:
            numText +=1
            print '处理文档数：',numText
            fileFullName = filepath_2+'/'+fileName
            # fileName = 'acq/0000005'
            # fileFullName = filePath+fileName
            tmpWord = []
            with open(fileFullName,'r') as f:
                texts = f.readlines()
                # print type(texts)

                sents = nltk.sent_tokenize("".join(texts))
                # print sents

                words = []
                for sent in sents:
                    words.append(nltk.word_tokenize(sent.strip()))

                for word in words:
                    for w in word:
                        ww =  w.lower().strip()
                        if ww not in english_punctuations and ww not in english_stopwords:
                            if re.match(r'^[a-z]+$',str(ww)):
                                tmpWord.append(st.stem(ww))
                # print tmpWord
            # ttword = [w for w in tmpWord if tmpWord.count(w)>1]
            trainSet.append(tmpWord)
            labels.append(fileType)
    print len(trainSet),len(labels)
    endTime = time.clock()
    print 'running time:',endTime-starttime
    return trainSet,labels

def buildVocablist(inputSet):
    """

    :param trainset:
    :return:
    """
    vocab = set([])
    for doc in inputSet:
        vocab = vocab | set(doc)

    return list(vocab)

def buildTfIdf(vocab,inputSet):
    """
    行为文档text，列为词表word
    :param vocab:
    :param inputSet:
    :return:
    """
    textNum = len(inputSet)
    vacLen = len(vocab)
    print textNum,vacLen
    tfMatrix = np.zeros([textNum,vacLen])
    print type(tfMatrix)
    for i in range(textNum):
        wordSet = set(inputSet[i])
        for w in wordSet:
            if w not in vocab:continue
            j = vocab.index(w)
            num = inputSet[i].count(w)
            tfMatrix[i][j] = num

    for i  in range(vacLen):
        numT = 0
        for wFre in tfMatrix[:,i]:
            if wFre!=0:
                numT+=1

        idf = np.log(textNum*1.0/(numT+1))

        tfMatrix[:,i] = tfMatrix[:,i]*idf
    return tfMatrix


def saveTfIdf(vocab,tfIdf):
    """

    :param tfIdf:
    :return:
    """
    with open('featureWord.txt','w') as f:
        f.write(",".join(vocab))
    with open('tfidf.txt','w') as f:
        for row in tfIdf:
            f.write(",".join([str(v) for v in row])+"\n")

def saveTfIdfEntro(vocab,tfIdf):
    """

    :param tfIdf:
    :return:
    """
    with open('featureWordentro.txt','w') as f:
        f.write(",".join(vocab))
    with open('tfidfentro.txt','w') as f:
        for row in tfIdf:
            f.write(",".join([str(v) for v in row])+"\n")

def loadTfIdfMat():
    """

    :return:
    """
    vocab=[]
    with open('featureWord.txt','r') as f:
        lines = f.readlines()
        lis = "".join(lines)
        vocab = lis.split(",")
    TfIdfMat = []
    with open('tfidf.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            TfIdfMat.append(line.split(","))
    return vocab,np.array(TfIdfMat)

def loadTfIdfMatEntro():
    """

    :return:
    """
    vocab=[]
    with open('featureWordentro.txt','r') as f:
        lines = f.readlines()
        lis = "".join(lines)
        vocab = lis.split(",")
    TfIdfMat = []
    with open('tfidfentro.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            TfIdfMat.append(line.split(","))
    return vocab,np.array(TfIdfMat)

def saveTrainSet(trainSet,labels):
    """

    :param trainSet:
    :param labels:
    :return:
    """
    with open('train.txt','w') as f:
        for i in range(len(trainSet)):
            f.write(",".join(trainSet[i])+","+labels[i]+"\n")

def loadTrainSet():
    """

    :return:
    """
    trainSet,labels = [],[]
    with open('train.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            trainSet.append(line.split(",")[:-1])
            labels.append(line.split(",")[-1].strip())
    return np.array(trainSet),labels

def loadTestSet():
    """
    :return:
    """
    trainSet,labels = [],[]
    with open('testSet.txt','r') as f:
        lines = f.readlines()
        for line in lines:
            trainSet.append(line.split(",")[:-1])
            labels.append(line.split(",")[-1].strip())
    return np.array(trainSet),labels

def loadTestCorpus():
    """

    :return:
    """
    starttime = time.clock()
    filePath = u'./test/'
    labels = []
    trainSet = []
    fileList = os.listdir(filePath)
    english_stopwords = stopwords.words('english')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','<','>','-']
    st = LancasterStemmer()
    # print english_stopwords
    numText = 0
    for fileType in fileList:
        filepath_2 = filePath+fileType
        fileList_2 = os.listdir(filepath_2)
        print '处理类别：',fileType
        for fileName in fileList_2:
            numText +=1
            print '处理文档数：',numText
            fileFullName = filepath_2+'/'+fileName
            # fileName = 'acq/0000005'
            # fileFullName = filePath+fileName
            tmpWord = []
            with open(fileFullName,'r') as f:
                texts = f.readlines()
                # print type(texts)

                sents = nltk.sent_tokenize("".join(texts))
                # print sents

                words = []
                for sent in sents:
                    words.append(nltk.word_tokenize(sent.strip()))

                for word in words:
                    for w in word:
                        if w.lower().strip() not in english_punctuations and w.lower().strip() not in english_stopwords:
                            if re.match(r'^[a-z]+$',str(w.lower().strip())):
                                tmpWord.append(st.stem(w.lower().strip()))
                # print tmpWord
            # ttword = [w for w in tmpWord if tmpWord.count(w)>1]
            trainSet.append(tmpWord)
            labels.append(fileType)
    print len(trainSet),len(labels)

    testSet = np.array(trainSet)
    labels = np.array(labels)
    endTime = time.clock()
    print 'running time:',endTime-starttime
    return testSet,labels


def buildbagofword(trainSet,vocab):
    """

    :param trainSet:
    :param vocab:
    :return:
    """
    textNum = len(trainSet)
    vacLen = len(vocab)
    print textNum, vacLen
    bowMat = np.zeros([vacLen, textNum])
    print type(bowMat)
    for i in range(textNum):
        wordSet = set(trainSet[i])
        for w in wordSet:
            j = vocab.index(w)
            num = trainSet[i].count(w)
            bowMat[j][i] = num

    return bowMat



def saveTestSet(testSet,labels):
    """

    :param trainSet:
    :param labels:
    :return:
    """
    with open('testSet.txt','w') as f:
        for i in range(len(testSet)):
            f.write(",".join(testSet[i])+","+labels[i]+"\n")

if __name__ == '__main__':
    trainSet,labels = readCorpus()
    saveTrainSet(trainSet,labels)
    vocab = buildVocablist(trainSet)
    tfIdfMat = buildTfIdf(vocab,trainSet)
    saveTfIdf(vocab,tfIdfMat)

