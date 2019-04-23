import numpy as np
from sklearn.utils import shuffle
from lib.ann import ann_1h
from lib.process import get_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def test_xor():
    # generate xor data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    ann_sigmoid = ann_1h('classification', size_i=X.shape[1], size_h=5, cls_set=set(Y))
    history_loss, history_acc = ann_sigmoid.fit(X, Y, learning_rate=0.01, verbose=True)

    print "final classification rate:", np.mean(ann_sigmoid.predict(X) == Y)
    plt.plot(history_acc)
    plt.show()

def test_donut():
    # generate donut data
    N = 1000
    R_inner = 5
    R_outer = 10
    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    # half points around inner radius:
    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    # half points around outer radius:
    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T
    # two classes: inner and outer
    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N/2) + [1]*(N/2))

    ann_sigmoid = ann_1h('classification', size_i=X.shape[1], size_h=8, cls_set=set(Y))
    history_loss, history_acc = ann_sigmoid.fit(X, Y, learning_rate=0.05, reg_rate=0.0002, verbose=True)

    plt.plot(history_acc)
    plt.show()


if __name__ == '__main__':
    test_xor()
    test_donut()
