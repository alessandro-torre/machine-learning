if __name__ == '__main__':

    from perceptron import Perceptron
    import numpy as np
    import datetime as dt
    import sys
    sys.path.append("../")
    from util import get_data as get_mnist
    from util import score

    X, Y = get_mnist('../data/train.csv', classes=np.array([5, 6]))  # import two classes
    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    perceptron = Perceptron(learning_rate=10e-3)
    t0 = dt.datetime.now()
    print 'Fitting..'
    perceptron.fit(Xtrain, Ytrain)
    t1 = dt.datetime.now()
    print 'Predicting (train set)..'
    Ytrain_perceptron = perceptron.predict(Xtrain)
    t2 = dt.datetime.now()
    print 'Predicting (test set)..'
    Ytest_perceptron = perceptron.predict(Xtest)
    t3 = dt.datetime.now()
    Strain_perceptron = score(Ytrain_perceptron, Ytrain)
    Stest_perceptron = score(Ytest_perceptron, Ytest)
    print 'Time to fit:', t1 - t0
    print 'Train set: score', Strain_perceptron, ', time', t2 - t1
    print 'Test set: score', Stest_perceptron, ', time', t3 - t2
    perceptron.plot()
