import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


def get_data(path, shuffle=True):
    # images are 48x48 = 2304 size vectors
    X = []
    Y = []
    first = True
    for line in open(path):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    X, Y = np.array(X) / 255.0, np.array(Y)
    if shuffle:
        idx = list(range(len(Y)))
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
    return X, Y


def show_im(X, Y):
    cls = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    while True:
        for i in range(7):
            x = X[Y == i]
            j = np.random.choice(len(x))
            n_pixel = int(np.sqrt(x.shape[1]))
            plt.imshow(x[j].reshape(n_pixel, n_pixel))
            plt.title(cls[i])
            plt.show()
        if input('Quit? (Y/n):') == 'Y':
            break


def main():
    print('importing fer data ...')
    X, Y = get_data('../large_data/fer/fer2013.csv', shuffle=True)
    #show_im(X, Y)
    print('fitting sequential model ...')
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(len(set(Y)), activation=tf.keras.activations.softmax)])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy()])
    result = model.fit(X, Y, validation_split=0.2, batch_size=500, epochs=100,
                       verbose=0)
    plt.plot(result.history['loss'], label='loss')
    plt.plot(result.history['accuracy'], label='accuracy')
    plt.plot(result.history['val_loss'], label='validation loss')
    plt.plot(result.history['val_accuracy'], label='validation accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
