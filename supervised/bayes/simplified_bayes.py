import numpy as np


class BinaryGaussianBayes(object):
    """This Bayes classifier is faster than a standard Bayes classifier, but restricted to binary classification.
    It also assumes that the features of a class are gaussian distributed.
    """
    def __init__(self):
        self.A = 0.
        self.w = 0.
        self.b = 0.
        self.d = 0  # number of characteristics
        self.classes = None
        self.trained = False

    def fit(self, x, y, smoothing=0.001):
        _, d = x.shape
        classes = np.unique(y)
        if len(classes) > 2:
            raise Exception("Not a binary dataset")
            pass
        else:
            c0 = (y == classes[0])
            c1 = (y == classes[1])
            x0 = x[c0, :]
            x1 = x[c1, :]
            logpriors = np.log(len(y[c1]) / float(len(y[c0])))
            mu0 = np.mean(x0, axis=0)
            mu1 = np.mean(x1, axis=0)
            cov0 = np.cov(x0.T) + smoothing * np.identity(d)
            cov1 = np.cov(x1.T) + smoothing * np.identity(d)
            invcov0 = np.linalg.inv(cov0)
            invcov1 = np.linalg.inv(cov1)
            det0overdet1 = np.linalg.det(cov0.dot(invcov1))  # together to avoid 0/0
            if det0overdet1 == 0:
                raise Exception("Covariance matrices with zero determinant")
                pass
            k = 0.5*np.log(det0overdet1)
            self.A = 0.5*(invcov0 - invcov1)
            self.w = invcov1.dot(mu1) - invcov0.dot(mu0)
            self.b = -0.5*mu1.T.dot(invcov1).dot(mu1) + 0.5*mu0.T.dot(invcov0).dot(mu0) + logpriors + k
            self.d = d
            self.classes = classes
            self.trained = True

    def predict(self, x):
        n, d = x.shape
        if not self.trained:
            raise Exception("The classifier has not been trained")
            pass
        elif d != self.d:
            raise Exception("Unmatched data dimension")
            pass
        else:
            y = np.repeat(self.classes[0], n)  # initialize to class 0
            is1 = np.sum((x.dot(self.A) + self.w) * x, axis=1) + self.b > 0  # decide if it is class 1
            y[is1] = self.classes[1]
            return y

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


if __name__ == '__main__':
    import sys
    sys.path.append("../")
    import util
    import datetime as dt

    from bayes import Bayes

    N = 10000
    X, Y = util.get_data('../data/train.csv', limit=N, classes=np.array([2, 5]), drop=True)  # import two classes
    N = len(Y)
    Ntrain = N / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    print '1. Standard Bayes method:'
    b = Bayes()
    t0 = dt.datetime.now()
    print 'Fitting..'
    b.fit(Xtrain, Ytrain)
    t1 = dt.datetime.now()
    print 'Predicting (train set)..'
    Ytrain_pred = b.predict(Xtrain)
    t2 = dt.datetime.now()
    print 'Predicting (test set)..'
    Ytest_pred = b.predict(Xtest)
    t3 = dt.datetime.now()
    Strain = util.score(Ytrain_pred, Ytrain)
    Stest = util.score(Ytest_pred, Ytest)
    print 'Time to fit:', t1 - t0
    print 'Train set: score', Strain, ', time', t2 - t1
    print 'Test set: score', Stest, ', time', t3 - t2

    print '2. Binary Gaussian Bayes:'
    sb = BinaryGaussianBayes()
    t0 = dt.datetime.now()
    print 'Fitting..'
    sb.fit(Xtrain, Ytrain)
    t1 = dt.datetime.now()
    print 'Predicting (train set)..'
    Ytrain_pred = sb.predict(Xtrain)
    t2 = dt.datetime.now()
    print 'Predicting (test set)..'
    Ytest_pred = sb.predict(Xtest)
    t3 = dt.datetime.now()
    Strain = util.score(Ytrain_pred, Ytrain)
    Stest = util.score(Ytest_pred, Ytest)
    print 'Time to fit:', t1 - t0
    print 'Train set: score', Strain, ', time', t2 - t1
    print 'Test set: score', Stest, ', time', t3 - t2
