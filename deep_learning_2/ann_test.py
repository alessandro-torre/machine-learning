import numpy as np
import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from lib.ann import ann


def main():
    # 3D saddle
    N = 1000
    X = np.random.random((N, 2)) * 4 - 2
    Y = X[:, 0] * X[:, 1]
    D = X.shape[1]

    # Test impact of additional layers
    if False:
        ann_set = {'simple': ann(n_features=D, hidden_layers_shape=5),
                   'deep': ann(n_features=D, hidden_layers_shape=[5, 5, 3])}
        history = {'simple': ann_set['simple'].fit(X, Y, learning_rate=0.01),
                   'deep': ann_set['deep'].fit(X, Y, learning_rate=0.01)}
        for ann_label in ann_set:
            plt.plot(history[ann_label]['loss'], label=ann_label)
        plt.legend()
        plt.show()

    # Test impact of sgd and momentum
    if False:
        ann_set = {'simple': ann(n_features=D, hidden_layers_shape=5),
                   'sgd': ann(n_features=D, hidden_layers_shape=5),
                   'momentum': ann(n_features=D, hidden_layers_shape=5),
                   'nesterov': ann(n_features=D, hidden_layers_shape=5),
                   'sgd+momentum': ann(n_features=D, hidden_layers_shape=5),
                   'sgd+nesterov': ann(n_features=D, hidden_layers_shape=5)}
        history = {'simple': ann_set['simple'].fit(X, Y, learning_rate=0.01, epochs=1000, batch_size=None, momentum=0.),
                   'sgd': ann_set['sgd'].fit(X, Y, learning_rate=0.01, epochs=1000, batch_size=100, momentum=0.),
                   'momentum': ann_set['momentum'].fit(X, Y, learning_rate=0.01, epochs=1000, batch_size=None,
                                                       momentum=0.99, nesterov=False),
                   'nesterov': ann_set['nesterov'].fit(X, Y, learning_rate=0.01, epochs=1000, batch_size=None,
                                                       momentum=0.99, nesterov=True),
                   'sgd+momentum': ann_set['sgd+momentum'].fit(X, Y, learning_rate=0.01, epochs=1000, batch_size=100,
                                                               momentum=0.99, nesterov=False),
                   'sgd+nesterov': ann_set['sgd+nesterov'].fit(X, Y, learning_rate=0.01, epochs=1000, batch_size=100,
                                                               momentum=0.99, nesterov=True)}
        for ann_label in ann_set:
            plt.plot(history[ann_label]['loss'], label=ann_label)
        plt.legend()
        plt.show()

    # Test impact of adaptive learning rates
    if True:
        ann_set = {'constant': ann(n_features=D, hidden_layers_shape=5),
                   'adagrad': ann(n_features=D, hidden_layers_shape=5),
                   'rmsprop': ann(n_features=D, hidden_layers_shape=5),
                   'adam': ann(n_features=D, hidden_layers_shape=5)}
        history = {'constant': ann_set['constant'].fit(X, Y, adaptive_learning='constant', learning_rate=0.01, epochs=1000, batch_size=100),
                   'adagrad': ann_set['adagrad'].fit(X, Y, adaptive_learning='adagrad', learning_rate=0.01, epochs=1000, batch_size=100),
                   'rmsprop': ann_set['rmsprop'].fit(X, Y, adaptive_learning='rmsprop', learning_rate=0.01, epochs=1000, batch_size=100),
                   'adam': ann_set['adam'].fit(X, Y, adaptive_learning='adam', learning_rate=0.01, epochs=1000, batch_size=100)}
        plt.plot(history['constant']['loss'], label='constant')
        plt.plot(history['adagrad']['loss'], label='adagrad')
        plt.plot(history['rmsprop']['loss'], label='rmsprop')
        plt.plot(history['adam']['loss'], label='adam')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
