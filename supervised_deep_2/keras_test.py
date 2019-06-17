import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from sklearn.model_selection import KFold


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
    result = {}
    for optimizer in optimizer_set:
        ann_set[optimizer] = ann_simple()
        ann_set[optimizer].compile(
            optimizer=optimizer_set[optimizer],
            loss='mse',
            metrics=['mae'])
        print("Training model", optimizer, "...")
        result[optimizer] = ann_set[optimizer].fit(
            X, Y, validation_split=0.2,
            epochs=100, batch_size=100, verbose=False)
        plt.plot(result[optimizer].history['val_loss'], label=optimizer)
    plt.legend()
    plt.title('Validation losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def main_crossval():
    """ Test cross validation. """
    X, Y = get_data()
    # Define splitting function
    kfold = KFold(n_splits=5, shuffle=True, random_state=1234)
    # Initialize model list and plots
    model = []
    ax_mse = plt.subplot(1, 2, 1, title='Validation mse', xlabel='epoch', ylabel='mse')
    ax_mae = plt.subplot(1, 2, 2, title='Validation mae', xlabel='epoch', ylabel='mae')
    # Cross-validation loop
    for idx_train, idx_valid in kfold.split(X, Y):
        model.append(ann_simple())
        model[-1].compile(
            optimizer=tf.train.AdamOptimizer(),
            loss='mse',
            metrics=['mae'])
        print("Training model", len(model), "...")
        model[-1].fit(
            X[idx_train], Y[idx_train],
            validation_data=(X[idx_valid], Y[idx_valid]),
            epochs=20, batch_size=100, verbose=0)
        val_mse = model[-1].history.history['val_loss']
        val_mae = model[-1].history.history['val_mean_absolute_error']
        print("    val_loss:", val_mse[-1])
        print("    val_mae :", val_mae[-1])
        ax_mse.plot(val_mse, label=len(model))
        ax_mae.plot(val_mae, label=len(model))
    ax_mse.legend(loc='upper right')
    ax_mae.legend(loc='upper right')
    plt.show()


def main_gridsearch():
    """ TODO: Try grid search.
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    """


if __name__ == '__main__':
    #main_adaptive()
    main_crossval()
    #main_gridsearch()
