import numpy as np
import pandas as pd

def get_data(binary=False):
    df = pd.read_csv('data/ecommerce_data.csv')
    N, D = df.shape

    #normalise
    df.loc[:,'n_products_viewed'] = (df['n_products_viewed'] - df['n_products_viewed'].mean()) / df['n_products_viewed'].std()
    df.loc[:,'visit_duration'] = (df['visit_duration'] - df['visit_duration'].mean()) / df['visit_duration'].std()
    
    #one-hot encoding for time_of_day
    col = 'time_of_day'
    NCat = len(set(df[col])) #NCat = len(df[col].unique())
    Z = np.zeros((N, NCat))
    Z[np.arange(N), df[col]] = 1
    df[[col + '_' + str(ii) for ii in np.arange(4)]] = pd.DataFrame(Z, index=df.index)
    df = df.drop(col, axis=1)
    
    #define Y (target) and X (input)
    Y = df['user_action'].values
    X = df[df.columns[df.columns!='user_action']].values

    #keep only classes 0,1
    if binary:
    	Y = Y[Y<2]
    	X = X[Y<2]

    return X, Y