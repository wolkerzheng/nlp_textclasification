#encoding=utf8
__author__ = 'ZGD'
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

class bayes:

    def __init__(self):
        """
        GaussianNB 它的假设是正态分布的.它的使用场景是.根据给定人物的高度和宽度,判定这个人的性别.
        分布分类中的词频不是正太分布
        MultinomialNB 他的假设就是出现次数.对于tf-idf向量的处理表现不错
        """
        #
        self.clf = MultinomialNB()

    def bayesfit(self,trainset,trainlable):

        self.clf.fit(trainset,trainlable)

    def bayesPredict(self,testset):

        return self.clf.predict(testset)


