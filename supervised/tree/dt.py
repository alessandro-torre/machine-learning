import numpy as np
import sys
sys.path.append("../")
import util


class BinaryTreeNode(object):
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth
        self.col = None  # on which feature (column of x) to split
        self.split = None  # at which value of the feature to do the split
        self.left = None  # left child TreeNode
        self.right = None  # right child TreeNode
        self.prediction = None  # prediction (make it a leaf node)
        self.trained = False

    def fit(self, x, y):
        classes = np.unique(y)
        if len(classes) > 2:
            raise Exception("Not a binary data set")
            pass
        else:
            if len(classes) == 1:
                self.prediction = classes[0]
                self.trained = True
            else:
                d = x.shape[1]
                cols = range(d)
                y_bin = 1*(y == classes[1])  # transform to binary classes 0, 1

                # determine the best split that maximizes ig
                max_ig = 0
                best_col = None  # best feature to split
                best_split = None  # best value to split at
                for col in cols:
                    # for each feature, find the best split and the associated information gain
                    split, ig = util.find_split(x[:, col], y_bin)
                    # compare the ig of this split with the maximum ig found so far
                    if ig > max_ig:
                        max_ig = ig
                        best_col = col
                        best_split = split

                if max_ig == 0:
                    # leaf node with no split (one prediction)
                    self.prediction = classes[np.round(y_bin.mean())]
                    self.trained = True
                else:
                    # node with split (may be leaf or root node)
                    self.col = best_col
                    self.split = best_split
                    left_idx = x[:, best_col] < best_split
                    right_idx = x[:, best_col] >= best_split

                    if self.depth == self.max_depth:
                        # leaf node with split (left and right predictions)
                        self.prediction = [
                            classes[np.round(y_bin[left_idx].mean())],
                            classes[np.round(y_bin[right_idx].mean())]
                        ]
                        self.trained = True
                    else:
                        # tree node with left and right children
                        self.left = BinaryTreeNode(self.depth + 1, self.max_depth)
                        self.right = BinaryTreeNode(self.depth + 1, self.max_depth)
                        self.left.fit(x[left_idx], y[left_idx])
                        self.right.fit(x[right_idx], y[right_idx])
                        self.trained = True

    def predict_one(self, x_row):
        if not self.trained:
            raise Exception("The node has not been trained")
            pass
        else:
            p = None
            if self.col is not None and self.split is not None:
                feature = x_row[self.col]
                if self.prediction is not None:
                    if feature < self.split:
                        p = self.prediction[0]
                    else:
                        p = self.prediction[1]
                elif self.left is not None and self.right is not None:
                    if feature < self.split:
                        p = self.left.predict_one(x_row)
                    else:
                        p = self.right.predict_one(x_row)
            else:
                p = self.prediction
            return p

    def predict(self, x):
        if not self.trained:
            raise Exception("The node has not been trained")
            pass
        else:
            n = len(x)
            p = np.zeros(n)
            for i in xrange(n):
                p[i] = self.predict_one(x[i])
            return p


class BinaryDecisionTree(object):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.trained = False

    def fit(self, x, y):
        self.root = BinaryTreeNode(max_depth=self.max_depth)
        self.root.fit(x, y)
        self.trained = True

    def predict(self, x):
        if not self.trained:
            raise Exception("The tree has not been trained")
            pass
        else:
            return self.root.predict(x)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)


if __name__ == '__main__':
    import datetime as dt
    sys.path.append("../bayes")
    from simplified_bayes import BinaryGaussianBayes

    X, Y = util.get_data('../../large_data/mnist/train.csv', classes=np.array([5, 6]))  # import two classes
    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    print '1. Binary Decision Tree:'
    tree = BinaryDecisionTree()
    t0 = dt.datetime.now()
    print 'Fitting..'
    tree.fit(Xtrain, Ytrain)
    t1 = dt.datetime.now()
    print 'Predicting (train set)..'
    Ytrain_tree = tree.predict(Xtrain)
    t2 = dt.datetime.now()
    print 'Predicting (test set)..'
    Ytest_tree = tree.predict(Xtest)
    t3 = dt.datetime.now()
    Strain_tree = util.score(Ytrain_tree, Ytrain)
    Stest_tree = util.score(Ytest_tree, Ytest)
    print 'Time to fit:', t1 - t0
    print 'Train set: score', Strain_tree, ', time', t2 - t1
    print 'Test set: score', Stest_tree, ', time', t3 - t2

    print '2. Binary Gaussian Bayes:'
    sb = BinaryGaussianBayes()
    t0 = dt.datetime.now()
    print 'Fitting..'
    sb.fit(Xtrain, Ytrain)
    t1 = dt.datetime.now()
    print 'Predicting (train set)..'
    Ytrain_bayes = sb.predict(Xtrain)
    t2 = dt.datetime.now()
    print 'Predicting (test set)..'
    Ytest_bayes = sb.predict(Xtest)
    t3 = dt.datetime.now()
    Strain_bayes = util.score(Ytrain_bayes, Ytrain)
    Stest_bayes = util.score(Ytest_bayes, Ytest)
    print 'Time to fit:', t1 - t0
    print 'Train set: score', Strain_bayes, ', time', t2 - t1
    print 'Test set: score', Stest_bayes, ', time', t3 - t2
