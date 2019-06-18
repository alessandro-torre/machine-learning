import numpy as np
import pandas as pd


def get_data(file_path, limit=None, classes=None, drop=False):
    print "Reading in and transforming data..."
    data = pd.read_csv(file_path).as_matrix()
    # shuffle before slicing
    np.random.shuffle(data)
    x = data[:, 1:] / 255.0  # data is from 0..255
    y = data[:, 0]
    # select given classes
    if classes is not None:
        # check that classes are valid
        if not all(np.in1d(classes, np.unique(y))):
            raise Exception('Invalid classes selected')
            pass
        else:
            select = np.in1d(y, classes)
            y = y[select]
            x = x[select, :]
            print 'Classes', classes, 'selected'
    if limit is not None:
        x, y = x[:limit], y[:limit]
        # dimensional reduction
        if drop:
            select = (x.sum(axis=0) > 0)
            x = x[:, select]  # remove zero columns
            print len(np.logical_not(select).nonzero()[0]), 'of', len(select), 'blank pixels removed'
    n, d = x.shape
    print n, 'images of', d, 'pixels loaded'
    return x, y


def get_xor():
    x = np.zeros((200, 2))
    x[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5-1, 0.5-1)
    x[50:100] = np.random.random((50, 2)) / 2  # (0-0.5, 0-0.5)
    x[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0-0.5, 0.5-1)
    x[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5-1, 0-0.5)
    y = np.array([0]*100 + [1]*100)
    # Shuffle data
    order = range(len(x))
    np.random.shuffle(order)
    return x[order], y[order]


def get_donut():
    n = 200
    r_inner = 5
    r_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    r1 = np.random.randn(n/2) + r_inner
    theta = 2*np.pi*np.random.random(n/2)
    x_inner = np.concatenate([[r1 * np.cos(theta)], [r1 * np.sin(theta)]]).T

    r2 = np.random.randn(n/2) + r_outer
    theta = 2*np.pi*np.random.random(n/2)
    x_outer = np.concatenate([[r2 * np.cos(theta)], [r2 * np.sin(theta)]]).T

    x = np.concatenate([ x_inner, x_outer ])
    y = np.array([0]*(n/2) + [1]*(n/2))
    return x, y


def score(y_pred, y):
        return np.mean(y_pred == y)


def entropy(y):
    """
    :param y: binary labels of a data set
    :type y: binary numpy array
    :return: the entropy of y
    :rtype: float
    """
    n = len(y)
    s1 = (y == 1).sum()
    if s1 == 0 or s1 == n:
        return 0
    p1 = float(s1) / n
    p0 = 1 - p1
    return - p0 * np.log2(p0) - p1 * np.log2(p1)


def information_gain(x, y, split):
    """
    :param x: values of a single feature of a data set
    :param y: binary labels of the data set
    :param split: where to split x
    :type x: numpy array
    :type y: binary numpy array
    :type split: float
    :return: the information gain on y when splitting x at split
    :rtype: float
    """
    y0 = y[x < split]
    y1 = y[x >= split]
    n = len(y)
    n0 = len(y0)
    if n0 == 0 or n0 == n:
        return 0
    p0 = float(n0) / n
    p1 = 1 - p0
    return entropy(y) - p0*entropy(y0) - p1*entropy(y1)


def find_split(x, y):
    """
    :param x: values taken by a feature of a data set
    :param y: (binary) labels associated to each value of x
    :type x: numpy array
    :type y: binary numpy array
    :return: the best split of x that maximizes the information gain on y, with the associated information gain
    :rtype: float, float
    """
    sort_idx = np.argsort(x)
    x_values = x[sort_idx]
    y_values = y[sort_idx]

    boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
    best_split = None
    max_ig = 0
    for i in boundaries:
        split = (x_values[i] + x_values[i+1]) / 2
        ig = information_gain(x_values, y_values, split)
        if ig > max_ig:
            max_ig = ig
            best_split = split
    return best_split, max_ig
