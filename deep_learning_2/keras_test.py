import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers

"""
TODO: add cross validation
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
TODO: add grid search
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
"""

def ann_simple():
    model = tf.keras.Sequential([
        layers.Dense(5, activation='relu'),
        layers.Dense(1)])
    return model


def get_data():
    """ Generate 3D saddle. """
    N = 1000
    X = np.random.random((N, 2)) * 4 - 2
    Y = X[:, 0] * X[:, 1]
    return X, Y


def main_adaptive():
    """ Test adaptive learning rates with Keras. """
    X, Y = get_data()
    optimizer_set = {
        'sgd': optimizers.SGD(lr=0.01),  # or tf.train.GradientDescentOptimizer
        'adagrad': optimizers.Adagrad(lr=0.01),  # or tf.train.AdagradOptimizer
        'rmsprop': optimizers.RMSprop(lr=0.01),  # or tf.train.RMSPropOptimizer
        'adam': optimizers.Adam(lr=0.01),  # or tf.train.AdamOptimizer
        'nadam': optimizers.Nadam(lr=0.01)}
    ann_set = {}
    history = {}
    for optimizer in optimizer_set:
        ann_set[optimizer] = ann_simple()
        ann_set[optimizer].compile(
            optimizer=optimizer_set[optimizer],
            loss='mse',
            metrics=['mae'])
        history[optimizer] = ann_set[optimizer].fit(X, Y, epochs=100)
        plt.plot(history[optimizer].history['loss'], label=optimizer)
    plt.legend()
    plt.show()


def main_crossval():
    """ TODO: Try cross validation.
https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    """


def main_gridsearch():
    """ TODO: Try grid search.
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    """


if __name__ == '__main__':
    #main_adaptive()
    main_crossval()
    #main_crossval()
    #main_gridsearch()
