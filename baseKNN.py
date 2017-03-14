#encoding=utf8
__author__ = 'ZGD'

import numpy

from sklearn.neighbors import KNeighborsClassifier

class knn:
    def __init__(self,n):
        self.neigh = KNeighborsClassifier(n_neighbors=n)

    def knnfit(self,x,y):
        self.neigh.fit(x,y)

    def knnPredict(self,x):
        return self.neigh.predict(x)

