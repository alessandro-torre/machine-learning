import os.path
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.stats import multivariate_normal as mvn


def get_mnist_data(folder: str, test_portion=0.1):
    df = pd.read_csv(os.path.join(folder, 'train.csv'))
    df = shuffle(df)
    Y = df['label'].values
    X = df[df.columns[df.columns!='label']].values / 255

    split = round(test_portion*len(df))
    Xtrain = X[:-split]
    Ytrain = Y[:-split]
    Xtest = X[split:]
    Ytest = Y[split:]

    return Xtrain, Ytrain, Xtest, Ytest


class GaussianNB(object):
    def __init__(self):
        self.classes = None
        self.prior = None
        self.mean  = None
        self.var   = None

    def train(self, X, Y, smoothing=1e-3):
        assert self.classes is None, 'Model already trained.'
        self.classes = dict(enumerate(set(Y)))
        K = X.shape[1]
        D = len(self.classes)
        self.prior = np.zeros(D)
        self.mean  = np.zeros((D, K))
        self.var   = np.zeros((D, K))
        for i, c in self.classes.items():
            x = X[Y==c]
            self.prior[i]  = len(x) / len(Y)
            self.mean[i,:] = x.mean(axis=0)
            self.var[i,:]  = x.var(axis=0) + smoothing

    def _predict_prob(self, X):
        assert self.classes is not None, 'Model not trained.'
        D, K = self.mean.shape
        N, KX = X.shape
        assert KX==K, 'Number of features in X not compatible with trained model.'
        prob = np.zeros((N, D))
        for i in self.classes.keys():
            prob[:, i] = np.log(self.prior[i]) \
                       + mvn.logpdf(X, mean=self.mean[i,:], cov=self.var[i,:])
        return prob

    def _predict_idx(self, X):
        return np.argmax(self._predict_prob(X), axis=1)

    def predict(self, X):
        idx = self._predict_idx(X)
        return list(map(self.classes.get, idx))

    def score(self, X, Y):
        assert set(Y).issubset(self.classes.values()), 'Some labels in Y are unknown to the model.'
        # Convert labels to indices
        lab2idx = dict(zip(self.classes.values(), self.classes.keys()))
        idx_true = np.array(list(map(lab2idx.get, Y)))
        # Predict and compare
        idx_pred = self._predict_idx(X)
        return (idx_pred==idx_true).mean()
