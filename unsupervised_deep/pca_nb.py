from lib.util import get_mnist_data
import numpy as np
import matplotlib.pyplot as plt

def pca(data, Q=None):
    if Q is None:
        # Eigendecomposition of the covariance matrix
        var, Q = np.linalg.eigh(np.cov(data.T))
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
        var = np.diag(np.cov(data_rotated))
        var = np.maximum(var, 0)  # may be slightly negative due to precision
        return data_rotated, var


def main():

    Xtrain, Ytrain, Xtest, Ytest = get_mnist_data(folder='../large_data/mnist/')

    Ztrain, trainVar, Q = pca(Xtrain)
    Ztest, testVar = pca(Xtest, Q)

    # Plot explained var and cumulative var
    plt.plot(trainVar)
    plt.title("Explained variance")
    plt.show()
    plt.plot(np.cumsum(trainVar))
    plt.title("Cumulative variance")
    plt.show()

    # Plot the first two dimensions
    plt.scatter(Ztrain[:,0], Ztrain[:,1], c=Ytrain)


    # TODO: naive bayes


if __name__=='__main__':
    main()
