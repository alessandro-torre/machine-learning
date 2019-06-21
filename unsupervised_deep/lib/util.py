import os.path
import pandas as pd
from sklearn.utils import shuffle

def get_mnist_data(folder: str, test_portion=0.1):
    df = pd.read_csv(os.path.join(folder, 'train.csv'))
    df = shuffle(df)
    Y = df['label'].values
    X = df[df.columns[df.columns!='label']].values / 255
    
    split = round(test_portion*len(df))
    Xtrain = X[:-split]
    Ytrain = Y[:-split]
    Xtest = X[split:]
    Ytest = Y[split:]
    
    return Xtrain, Ytrain, Xtest, Ytest
