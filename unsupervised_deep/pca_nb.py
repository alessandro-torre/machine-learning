import numpy as np
import time
import lib.util
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt


def do_bayes(Xtrain, Ytrain, Xtest, Ytest):
    t0 = time.time()
    nb = lib.util.GaussianNB()
    nb.train(Xtrain, Ytrain)
    print('Train score: ', nb.score(Xtrain, Ytrain))
    print('Test  score: ', nb.score(Xtest, Ytest))
    print('Total time : ', time.time()-t0)


def main():

    Xtrain, Ytrain, Xtest, Ytest = lib.util.get_mnist_data(folder='../large_data/mnist/')

    # Naive bayes without PCA
    print('Doing naive bayes classification on all features..')
    do_bayes(Xtrain, Ytrain, Xtest, Ytest)

    # Do PCA on train data, and apply the same to test data
    print('Doing PCA on train data..')
    pca = lib.decomposition.PCA()
    t0 = time.time()
    Ztrain = pca.train_and_transform(Xtrain)
    Ztest  = pca.transform(Xtest)
    print('Time to do PCA:', time.time()-t0)

    # Naive bayes with PCA
    n_features = 10
    pca_var_n = pca.var[:n_features].sum() / pca.var.sum()
    print(f'The first {n_features} features of PCA ' +
          f'explain {(int)(round(pca_var_n*100))}% of variance.')
    print('Doing naive bayes classification on these features..')
    Ztrain_ = Ztrain[:,:n_features]
    Ztest_  = Ztest[:,:n_features]
    do_bayes(Ztrain_ , Ytrain, Ztest_, Ytest)

    # Naive bayes with PCA (keep more features)
    n_features = 50
    pca_var_n = pca.var[:n_features].sum() / pca.var.sum()
    print(f'The first {n_features} features of PCA ' +
          f'explain {(int)(round(pca_var_n*100))}% of variance.')
    print('Doing naive bayes classification on these features..')
    Ztrain_ = Ztrain[:,:n_features]
    Ztest_  = Ztest[:,:n_features]
    do_bayes(Ztrain_ , Ytrain, Ztest_, Ytest)


if __name__=='__main__':
    main()
