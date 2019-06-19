import os.path
import pandas as pd
from sklearn.utils import shuffle

def get_mnist_data(folder: str):
    train = pd.read_csv(os.path.join(folder, 'train.csv'))
    train = shuffle(train)
    Ytrain = train['label'].values
    Xtrain = train[train.columns[train.columns!='label']].values / 255

    test = pd.read_csv(os.path.join(folder, 'test.csv'))
    Ytest = test['label'].values
    Xtest = test[test.columns[test.columns!='label']].values / 255
       
    return Xtrain, Ytrain, Xtest, Ytest
