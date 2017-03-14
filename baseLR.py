#encoding=utf8
__author__ = 'ZGD'

from sklearn.linear_model import LogisticRegression


class lr:

    def __init__(self,c=1e5):
        self.logreg = LogisticRegression(C=c)

    def lrfit(self,x,y):
        self.logreg.fit(x,y)

    def lrpredict(self,x):
        return self.logreg.predict(x)
