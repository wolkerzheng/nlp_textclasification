#encoding=utf8
__author__ = 'ZGD'
from sklearn.svm import SVC

class  mySVM:

    def __init__(self,type='rbf'):
        self.clf = SVC(kernel=type)

    def svmFit(self,trainSet,labels):

        self.clf.fit(trainSet,labels)

    def svmPredict(self,testSet):

        return self.clf.predict(testSet)