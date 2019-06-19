import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sortedcontainers import SortedList


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x  in enumerate(X):
            # knn is the sorted list of k nearest neighbours
            # each element of knn is a tuple (distance, class)
            # the list is sorted according to distance
            knn = SortedList(load=self.k)
            for j, self_x in enumerate(self.X):
                d = np.linalg.norm(x - self_x)
                knn.add((d, self.y[j]))
                if len(knn) > self.k:
                    del knn[-1]
            # assign to X[i] the mode of knn classes
            y[i] = sp_stats.mode([el[1] for el in knn])[0][0]
            # # knn_freq is a dictionary with key=class, value=count (occurrence)
            # knn_freq = {}
            # for _, cl in knn:
            #     knn_freq[cl] = knn_freq.get(cl, 0) + 1
            # # assign a class to X[i] based on most frequent class in knn_freq
            # y[i] = max(knn_freq)
        return y

    def score_y(self, y_pred, y):
        return np.mean(y_pred == y)

    def score(self, X, y):
        return self.score_y(self.predict(X), y)


if __name__ == '__main__':

    from util import get_data
    from datetime import datetime
    from matplotlib import pyplot as plt

    X, Y = get_data('../../large_data/mnist/train.csv', limit=1000)

    # check one image
    n = 3
    print "Label of element", n, ":", Y[n]
    plt.imshow(X[n].reshape(28, 28))
    plt.show()

    Ntrain = 500
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    for k in (2,3):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print "Training time:", (datetime.now() - t0)
        print "Predicted label of element", n, ":", knn.predict(X[n].reshape(1,784))[0]

        t0 = datetime.now()
        print "Train accuracy:", knn.score(Xtrain, Ytrain)
        print "Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain)

        t0 = datetime.now()
        print "Test accuracy:", knn.score(Xtest, Ytest)
        print "Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest)
