import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from lib.ann import ann


def get_data(path, shuffle=True):
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
            x = X[Y == i, :]
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
    show_im(X, Y)
    print('fitting ann ...')
    ann1 = ann(n_features=X.shape[1], classification_set=set(Y), hidden_layers_shape=[50, 20], activation='tanh')
    history1 = ann1.fit(X[:-1000], Y[:-1000], Xvalid=X[-1000:], Yvalid=Y[-1000:],
                        learning_rate=0.1, momentum=0.90, adaptive_learning='constant',
                        epochs=100, batch_size=500, verbose=True)
    plt.plot(history1['loss'], label='loss')
    plt.plot(history1['metric'], label='accuracy')
    plt.plot(history1['loss_v'], label='validation loss')
    plt.plot(history1['metric_v'], label='validation accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
