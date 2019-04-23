import numpy as np
import scipy.stats as sp_stats


class Bayes(object):
    """Standard Bayes classifier with gaussian assumption on the distribution of the features of a class."""
    def __init__(self):
        self.gaussians = dict()
        self.logpriors = dict()
        self.d = 0
        self.K = 0
        self.trained = False

    def fit(self, x, y, smoothing=0.001):
        _, self.d = x.shape
        self.classes = np.unique(y)
        for c in self.classes:
            x_c = x[y == c]
            self.logpriors[c] = np.log(len(y[y == c]) / float(len(y)))
            self.gaussians[c] = {
                'mean': np.mean(x_c, axis=0),
                'cov': np.cov(x_c.T) + smoothing * np.identity(self.d)
            }
        self.K = len(self.classes)
        self.trained = True

    def predict(self, x):
        n, d = x.shape
        if not self.trained:
            raise Exception("Unfitted model")
            pass
        elif d != self.d:
            raise Exception("Unmatched data dimension")
            pass
        else:
            p = np.zeros((n, self.K))
            for i in xrange(self.K):
                c = self.classes[i]
                p[:, i] = sp_stats.multivariate_normal.logpdf(x, self.gaussians[c]['mean'], self.gaussians[c]['cov']) \
                          + self.logpriors[c]
            select = np.argmax(p, axis=1)
            return self.classes[select.T]

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


if __name__ == '__main__':
    import sys
    sys.path.append("../")
    import util
    import datetime as dt

    N = 10000
    X, Y = util.get_data('../data/train.csv', limit=N)
    N = len(Y)
    Ntrain = N / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

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
