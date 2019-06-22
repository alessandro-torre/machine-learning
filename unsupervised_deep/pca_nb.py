import numpy as np
import time
from scipy.stats import multivariate_normal as mvn
import lib.util
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt

def pca(data, Q=None):
    if Q is None:
        # Eigendecomposition of the covariance matrix
        var, Q = np.linalg.eigh(np.cov(data, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision
        var = np.maximum(var, 0)
        # Order features from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        Q = Q[:, idx]
        # Rotate the data to diagonalize its covariance
        data_rotated = data.dot(Q)
        return data_rotated, var, Q
    else:
        # Rotate the data with given Q
        assert data.shape[1]==Q.shape[0], 'Incompatible dimension of data and Q.'
        data_rotated = data.dot(Q)
        var = np.diag(np.cov(data_rotated, rowvar=False))
        # Some eigenvalues may be slightly negative due to machine precision
        var = np.maximum(var, 0)
        return data_rotated, var


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


def main():

    Xtrain, Ytrain, Xtest, Ytest = lib.util.get_mnist_data(folder='../large_data/mnist/')

    # Do pca on train data, and apply the same to test data
    print('Doing PCA on train data..')
    Ztrain, ZtrainVar, Q = pca(Xtrain)
    Ztest, _ = pca(Xtest, Q)

    # Plot: explained var and cumulative var
    plt.plot(ZtrainVar/ZtrainVar.sum())
    plt.title('Explained variance')
    plt.show()
    plt.plot(np.cumsum(ZtrainVar)/ZtrainVar.sum())
    plt.title('Cumulative variance')
    plt.show()

    # How much variance do we explain with the first 2 variables?
    ZtrainVar_2 = ZtrainVar[:2].sum() / ZtrainVar.sum()
    print(f'The first 2 vars explain {(int)(round(ZtrainVar_2*100))}% of variance.')

    # How many variables we need to keep to explain a target variance?
    alpha = [0.50, 0.90, 0.95]
    for al in alpha:
        keep = np.searchsorted(np.cumsum(ZtrainVar), al*ZtrainVar.sum())
        print(f'Keep first {keep}/{len(ZtrainVar)} features to explain {(int)(round(al*100))}% of variance.')

    # Plot the first two dimensions
    plt.scatter(Ztrain[:,0], Ztrain[:,1], c=Ytrain)
    plt.title('Data clustering by first two features.')
    plt.show()

    # Naive bayes without pca
    print('Doing naive bayes classification on all features..')
    t0 = time.time()
    nb = GaussianNB()
    nb.train(Xtrain, Ytrain)
    print('Train score: ', nb.score(Xtrain, Ytrain))
    print('Test score : ', nb.score(Xtest, Ytest))
    print('Total time :', time.time()-t0)

    # Naive bayes with pca
    n_features = 10
    ZtrainVar_n = ZtrainVar[:n_features].sum() / ZtrainVar.sum()
    print(f'The first {n_features} features of PCA ' +
          f'explain {(int)(round(ZtrainVar_n*100))}% of variance.')
    print('Doing naive bayes classification on these features..')
    t0 = time.time()
    Ztrain_ = Ztrain[:,:n_features]
    Ztest_  = Ztest[:,:n_features]
    nbPca = GaussianNB()
    nbPca.train(Ztrain_, Ytrain)
    print('Train score: ', nbPca.score(Ztrain_, Ytrain))
    print('Test  score: ', nbPca.score(Ztest_, Ytest))
    print('Total time :', time.time()-t0)

    # Naive bayes with pca (more features)
    n_features = 50
    ZtrainVar_n = ZtrainVar[:n_features].sum() / ZtrainVar.sum()
    print(f'The first {n_features} features of PCA ' +
          f'explain {(int)(round(ZtrainVar_n*100))}% of variance.')
    print('Doing naive bayes classification on these features..')
    t0 = time.time()
    Ztrain_ = Ztrain[:,:n_features]
    Ztest_  = Ztest[:,:n_features]
    nbPca = GaussianNB()
    nbPca.train(Ztrain_, Ytrain)
    print('Train score: ', nbPca.score(Ztrain_, Ytrain))
    print('Test  score: ', nbPca.score(Ztest_, Ytest))
    print('Total time :', time.time()-t0)


if __name__=='__main__':
    main()
