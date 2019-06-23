import numpy as np
import random
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

    # Reduce the number of samples, just to test. Naive implementation of SVD too slow
    idx = random.randint(0, len(Ytrain))
    # TODO: use idx
    #Xtrain = Xtrain[:1000]
    #Ytrain = Ytrain[:1000]

    # Do SVD on train data. Keep 2 features and get transformed data.
    # Note: with SVD, you cannot reduce features after transforming.
    print('Doing SVD on train data..')
    svd = lib.decomposition.SVD()
    Ztrain = svd.train_and_transform(Xtrain, n_features=2)

    # Plot: explained var and cumulative var
    plt.plot(svd.var/svd.var.sum())
    plt.title('Explained variance (%)')
    plt.show()
    plt.plot(np.cumsum(svd.var)/svd.var.sum())
    plt.title('Cumulative variance (%)')
    plt.show()

    # How much variance do we explain with the first 2 variables?
    svd_var_2 = svd.var[:2].sum() / svd.var.sum()
    print(f'The first 2 vars explain {(int)(round(svd_var_2*100))}% of variance.')

    # How many variables do we need to explain a target variance?
    alpha = [0.50, 0.90, 0.95]
    for al in alpha:
        keep = np.searchsorted(np.cumsum(svd.var), al*svd.var.sum())
        print(f'Keep first {keep}/{len(svd.var)} features to explain {(int)(round(al*100))}% of variance.')

    # Plot the transformed data
    print(Ztrain.shape)
    plt.scatter(Ztrain[:,0], Ztrain[:,1], c=Ytrain)
    plt.title('Data clustering by first two features.')
    plt.show()

    # TODO: Xtrain_=svd.approximate(Xtrain, n_features=2) and check Frobenius error of reconstruction

if __name__=='__main__':
    main()
