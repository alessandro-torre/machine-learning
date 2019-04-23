import numpy as np
import scipy.stats as sp_stats

class NaiveBayes(object):
    def __init__(self):
        self.gaussians = dict()
        self.logpriors = dict()
        self.K = 0

    def fit(self, X, Y, smoothing=0.001):
        classes = set(Y)
        self.K = len(classes)
        for c in classes:
            X_c = X[Y == c]
            self.logpriors[c] = np.log(len(Y[Y == c]) / float(len(Y)))
            self.gaussians[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + smoothing
            }

    def predict(self, X):
        nX, _ = X.shape
        p = np.zeros((nX, self.K))
        for c, g in self.gaussians.iteritems():
            p[:,c] = sp_stats.multivariate_normal.logpdf(X, g['mean'], g['var']) \
                + self.logpriors[c]
        return np.argmax(p, axis=1)

    @staticmethod
    def score_y(Y_pred, Y):
        return np.mean(Y_pred == Y)

    def score(self, X, Y):
        return self.score_y(self.predict(X), Y)


if __name__ == '__main__':

    import sys
    sys.path.append("../")
    import util
    import datetime as dt

    N = 10000
    Ntrain = N / 2
    X, Y = util.get_data('../data/train.csv', limit=N)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    nb = NaiveBayes()
    t0 = dt.datetime.now()
    print 'Fitting..'
    nb.fit(Xtrain, Ytrain)
    t1 = dt.datetime.now()
    print 'Predicting (train set)..'
    Ytrain_pred = nb.predict(Xtrain)
    t2 = dt.datetime.now()
    print 'Predicting (test set)..'
    Ytest_pred = nb.predict(Xtest)
    t3 = dt.datetime.now()
    Strain = nb.score_y(Ytrain_pred, Ytrain)
    Stest = nb.score_y(Ytest_pred, Ytest)

    print 'Time to fit:', t1-t0
    print 'Train set: score', Strain, ', time', t2-t1
    print 'Test set: score', Stest, ', time', t3-t2
