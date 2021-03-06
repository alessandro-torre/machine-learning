from knn import KNN
from util import get_xor
import matplotlib.pyplot as plt

if __name__=='__main__':

    X, Y = get_xor()
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y, alpha=0.5)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    Y_pred = model.predict(X)
    print "Accuracy:", model.score_y(Y_pred, Y)
    plt.scatter(X[:, 0], X[:, 1], s=100, c=Y_pred, alpha=0.5)
    plt.show()
