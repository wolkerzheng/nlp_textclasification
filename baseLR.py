#encoding=utf8
__author__ = 'ZGD'

from sklearn.linear_model import LogisticRegression
import numpy as np

class lr:

    def __init__(self,c=1e5):
        self.logreg = LogisticRegression(C=c)

    def lrfit(self,x,y):
        self.logreg.fit(x,y)

    def lrpredict(self,x):
        return self.logreg.predict(x)

class myLr:
    """
    实现逻辑回归模型
    """
    def __init__(self,alpha=0.01,maxCycles = 100):
        """
        初始化参数,alpha为步长（学习率）；maxCycles最大迭代次数
        :param alpha:
        :param maxCycles:
        """
        self.alpha = alpha
        self.maxCycles = maxCycles

    def sigmoid(self,x):
        """
        sigmoid函数: f(x) = 1 / (1+e^{-x})
        :param x:
        :return:
        """
        return 1.0/(1+np.exp(-x))


    def softmax(self,x):
        """

        :param x:
        :return:
        """

        return (1-np.exe(-x))/(1+np.exe(-x))


    def gradDescent(self,x,y):
        """

        :param x:
        :param y:
        :return:
        """
        dataMat = np.mat(x)  # size: m*n
        labelMat = np.mat(y).transpose()  # size: m*1
        m, n = np.shape(x)
        weights = np.ones((n, 1))
        for i in xrange(0, self.maxCycles):

            #批量更新权值
            hx = self.sigmoid(dataMat*weights)
            error = labelMat - hx
            weights = weights + self.alpha*dataMat.transpose() * error
        return weights

    def fit(self,x,y):
        """

        :param x:
        :param y:
        :return:
        """
        if not isinstance(x,np.ndarray):
            x = np.array(x)

        if not isinstance(y,np.ndarray):
            y = np.array(y)

        return self.gradDescent(x,y)

    # 使用学习得到的参数进行分类
    def predict(self, test_X, weigh):
        """

        :param test_X:
        :param weigh:
        :return:
        """
        dataMat = np.mat(test_X)

        hx = self.sigmoid(dataMat * weigh)  # size:m*1
        m = len(hx)
        predict = []
        for i in range(m):
            if int(hx[i]) > 0.5:
                predict.append(1)
            else:
                predict.append(0)
        return predict

