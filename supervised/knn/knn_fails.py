import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

## create 2d grid of points with alternate labels (0,1)
def get_data():
    width = 8
    height = 8
    N = width * height

    # first column is label Y, second two columns are pixel's coordinates
    data = np.zeros((N, 3))
    n = 0
    start_t = 0 # starting label of first point
    for i in xrange(width):
        t = start_t
        for j in xrange(height):
            data[n, 0] = t # assign label
            data[n, 1:] = [i, j] # assign coordinates
            n += 1
            t = (t + 1) % 2 # flip label of next column
        start_t = (start_t + 1) % 2 # flip starting label of next row
    # shuffle in order to select sparse points if Ntrain < N
    np.random.shuffle(data)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

if __name__ == '__main__':
    X, Y = get_data()

    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    model = KNN(3)
    Ntrain = len(Y)
    model.fit(X[:Ntrain, :], Y[:Ntrain])
    print "Train accuracy:", model.score(X, Y)

    Y_pred = model.predict(X)
    plt.scatter(X[:,0], X[:,1], s=100, c=Y_pred, alpha=0.5)
    plt.show()
