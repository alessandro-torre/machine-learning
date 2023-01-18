import numpy as np
import lib.util
import lib.decomposition
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt


def main():

    Xtrain, Ytrain, Xtest, Ytest = lib.util.get_mnist_data(folder='../large_data/mnist/')

    # Do PCA on train data.
    print('Doing PCA on train data..')
    pca = lib.decomposition.PCA()
    Ztrain = pca.train_and_transform(Xtrain)

    # Calculate the reconstruction error (Frobenius norm) when keeping 2 features
    Xtrain_ = pca.approximate(Xtrain, n_features=2)
    error = np.sqrt(np.sum((Xtrain - Xtrain_)**2)) / len(Xtrain)
    print(f'Reconstruction error with two features: {error}')

    # How much variance do we explain with the first 2 variables?
    pca_var_2 = pca.var[:2].sum() / pca.var.sum()
    print(f'The first 2 features explain {(int)(round(pca_var_2*100))}% of variance.')

    # How many variables do we need to explain a target variance?
    alpha = [0.50, 0.90, 0.95]
    for al in alpha:
        keep = np.searchsorted(np.cumsum(pca.var), al*pca.var.sum())
        print(f'Keep first {keep}/{len(pca.var)} features to explain {(int)(round(al*100))}% of variance.')

    # Plot: explained var and cumulative var
    plt.plot(pca.var/pca.var.sum())
    plt.title('Explained variance (%)')
    plt.show()
    plt.plot(np.cumsum(pca.var)/pca.var.sum())
    plt.title('Cumulative variance (%)')
    plt.show()

    # Plot the first two features of the transformed data
    plt.scatter(Ztrain[:,0], Ztrain[:,1], c=Ytrain)
    plt.title('Data clustering by first two features.')
    plt.show()


if __name__=='__main__':
    main()
