import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, learning_rate=1.0, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epochs = None  # epochs actually used when fitting
        self.costs = []  # evolution of the misclassification rate
        self.w = None
        self.b = None
        self.classes = None
        self.trained = False

    def fit(self, x, y):
        classes = np.unique(y)
        if len(classes) != 2:
            raise Exception("Not a binary data set")
            pass
        else:
            n, d = x.shape  # number of data points and of features

            # transform to binary classes -1, +1
            y_bin = -1 * (y == classes[0]) + 1 * (y == classes[1])

            # initialize fit parameters
            self.w = np.random.randn(d)
            self.b = 0
            self.classes = classes
            self.trained = True

            for epoch in xrange(self.max_epochs):
                self.epochs = epoch + 1
                # for each epoch, do a prediction with the latest w and b
                y_hat = self.predict(x)
                incorrect = np.nonzero(y != y_hat)[0]
                if len(incorrect) == 0:
                    break
                # randomly pick up a misclassified point and update w and b
                i = np.random.choice(incorrect)
                self.w += self.learning_rate*y_bin[i]*x[i]
                self.b += self.learning_rate*y_bin[i]
                # update the history of the misclassification rate
                c = float(len(incorrect)) / n
                self.costs.append(c)

    def plot(self):
        if not self.trained:
            raise Exception("The perceptron has not been trained")
            pass
        else:
            print "final w:", self.w, "final b:", self.b, "epochs:", self.epochs, "/", self.max_epochs
            plt.plot(self.costs)
            plt.show()

    def predict(self, x):
        if not self.trained:
            raise Exception("The perceptron has not been trained")
            pass
        else:
            p = np.sign(x.dot(self.w) + self.b)  # predictions as {-1, +1}
            return self.classes[1*(p==1)]  # prediction as classes

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


if __name__ == '__main__':
    import datetime as dt
    import sys
    sys.path.append("../")
    from util import score

    def get_data():
        """Generate data with an underlying perceptron model"""
        w = np.array([-0.5, 0.5])  # normal vector of [0.5, 0.5]
        b = 0.1
        x = np.random.random((300, 2)) * 2 - 1  # 2d vectors
        y = np.sign(x.dot(w) + b)  # labels are {-1, +1}
        return x, y

    X, Y = get_data()
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
    plt.show()

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

    # overview of classified test points (darker if misclassified)
    colors = Ytest_perceptron*(1+10*(Ytest_perceptron != Ytest))
    plt.scatter(Xtest[:, 0], Xtest[:, 1], c=colors, s=100, alpha=0.5)
    plt.show()
