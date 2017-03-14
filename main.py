#encoding=utf8
__author__ = 'ZGD'

import numpy as np
import baseBayes
import featureExtraction
import baseKNN
import baseLR
import baseSVM
import process
from sklearn.metrics import  classification_report,confusion_matrix

if __name__ == '__main__':

    #加载训练集
    # trainSet,labels = process.readCorpus()
    # process.saveTrainSet(trainSet,labels)
    trainSet,labels = process.loadTrainSet()
    vocab = process.buildVocablist(trainSet)
    bowMat = process.buildbagofword(trainSet,vocab)
    #信息增益
    wordEntDict = featureExtraction.calculateEntropy(bowMat,labels,vocab)
    featureVocab = featureExtraction.selectBestKByEntropy(wordEntDict,k=10000)
    tfIdfMat = process.buildTfIdf(featureVocab, trainSet)
    # featureVocab, tfIdfMat = process.loadTfIdfMatEntro()
    #chi2检验
    # wordChiDict = featureExtraction.calculateChi(bowMat,labels,vocab)
    # featword = featureExtraction.selectBestKByChi(vocab,wordChiDict)
    # tfIdfMat = process.buildTfIdf(featureVocab,trainSet)
    # process.saveTfIdfEntro(featureVocab,tfIdfMat)
    # featword, tfIdfMat = process.loadTfIdfMat()

    # featureVocab,tfIdfMat = process.loadTfIdfMatEntro()


    # 加载并保存测试集
    # testSet,testLabel = process.loadTestCorpus()
    # process.saveTestSet(testSet,testLabel)
    testSet, testLabel = process.loadTestSet()
    testtfIdmMat = process.buildTfIdf(featureVocab,testSet)

    print type(tfIdfMat)
    svm = baseSVM.mySVM()
    svm.svmFit(tfIdfMat, labels)
    y_predict = svm.svmPredict(testtfIdmMat)

    print 'predict:',np.mean(y_predict==testLabel)
    print 'precision,recall,F1'
    print classification_report(testLabel,y_predict)

    # bay = baseBayes.bayes()
    # bay.bayesfit(tfIdfMat, labels)
    # y_predict = bay.bayesPredict(testtfIdmMat)
    # kn = baseKNN.knn(n=25)
    # kn.knnfit(tfIdfMat, labels)
    # y_predict = kn.knnPredict(testtfIdmMat)
    # lrr = baseLR.lr(c=1e6)
    # lrr.lrfit(tfIdfMat, labels)
    # y_predict = lrr.lrpredict(testtfIdmMat)
    # svm = baseSVM.mySVM()
    # svm.svmFit(tfIdfMat, labels)
    # y_predict = svm.svmPredict(testtfIdmMat)