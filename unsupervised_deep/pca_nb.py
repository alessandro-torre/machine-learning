from lib.util import get_mnist_data
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

def pca(data, Q=None):
    if Q is None:
        # Eigendecomposition of the covariance matrix
        var, Q = np.linalg.eigh(np.cov(data, rowvar=False))
        var = np.maximum(var, 0)  # may be slightly negative due to precision
        # Order features from most to least significant
        idx = np.argsort(-var)
        var = var[idx]
        Q = Q[:, idx]
        # Rotate the data to diagonalize its covariance
        data_rotated = data.dot(Q)
        return data_rotated, var, Q
    else:
        # Rotate the data with given Q
        data_rotated = data.dot(Q)
        var = np.diag(np.cov(data_rotated, rowvar=False))
        var = np.maximum(var, 0)  # may be slightly negative due to precision
        return data_rotated, var


class GaussianNB(object):
    def __init__(self):
        self.classes = None
        self.prior = dict()
        self.mean  = dict()
        self.var   = dict()
    
    def train(self, X, Y):
        assert self.classes is None, 'Model already trained.'
        self.classes = set(Y)   
        for c in self.classes:
            x = X[Y==c]
            self.prior[c] = len(x) / len(Y)
            self.mean[c]  = x.mean()
            self.var[c]   = x.var()
    
    def predict(self, X):
        assert self.classes is not None, 'Model not trained.'
        N, _ = X.shape
        K = len(self.classes)
        prob = np.zeros((N, K))
        for c in self.classes:
            prob[:, c] = np.log(self.prior[c]) 
            + mvn.logpdf(X, mean=self.mean[c], cov=self.var[c])
        return self.classes[np.argmax(prob, axis=1)]

    def score(self, X, Y):
        assert set(Y).issubset(self.classes), 'Some labels were not seen at training time.'
        Yhat = self.predict(X)
        return (Yhat==Y).mean()


def main():

    Xtrain, Ytrain, Xtest, Ytest = get_mnist_data(folder='../large_data/mnist/')
    
    # Do pca on train data, and apply the same to test data
    Ztrain, trainVar, Q = pca(Xtrain)
    Ztest, testVar = pca(Xtest, Q)

    # Plot: explained var and cumulative var
    plt.plot(trainVar)
    plt.title("Explained variance")
    plt.show()
    plt.plot(np.cumsum(trainVar))
    plt.title("Cumulative variance")
    plt.show()

    # How much variance do we explain with the first 2 variables?
    trainVar_firstTwo = trainVar[0:2].sum() / trainVar.sum()
    print(f'The first 2 vars explain {(int)(round(trainVar_firstTwo*100))}% of variance.')
    
    # How many variables we need to keep to explain a target variance?
    alpha = [0.50, 0.90, 0.95]
    for al in alpha:
        keep = np.searchsorted(np.cumsum(trainVar), al*trainVar.sum())
        print(f'Keep {keep}/{len(trainVar)} vars to explain {(int)(round(al*100))}% of variance.')

    # Plot the first two dimensions
    plt.scatter(Ztrain[:,0], Ztrain[:,1], c=Ytrain) 
    
    # Naive bayes without pca
    nb = GaussianNB()
    nb.train(Xtrain, Ytrain)
    print('Train score: ', nb.score(Xtrain, Ytrain))
    print('Test  score: ', nb.score(Xtest, Ytest))
    
    # Naive bayes with pca
    nbPca = GaussianNB()
    nbPca.train(Ztrain, Ytrain)
    print('Train score: ', nb.score(Ztrain, Ztrain))
    print('Test  score: ', nb.score(Ztest, Ztest))

if __name__=='__main__':
    main()
