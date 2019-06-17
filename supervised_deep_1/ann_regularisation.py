import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from lib.ann import ann

# 3D saddle
N = 1000
X = np.random.random((N, 2))*4 - 2
Y = X[:,0]*X[:,1]
D = X.shape[1]

Ntrain = 500
Xtrain = X[:Ntrain, :]
Ytrain = Y[:Ntrain]
Xtest = X[Ntrain:, :]
Ytest = Y[Ntrain:]

ann_1 = ann(n_features=D, n_hidden_layers=2, hidden_layers_size=10)
ann_reg = ann(n_features=D, n_hidden_layers=2, hidden_layers_size=10)

loss_train, _, loss_test, _ = ann_1.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.01, epochs=10000)
loss_train_reg, _, loss_test_reg, _ = ann_reg.fit(Xtrain, Ytrain, Xtest, Ytest, learning_rate=0.01, epochs=10000, reg_rate=0.01)


plt.plot(loss_train, label='train')
plt.plot(loss_test, label='test')
plt.plot(loss_train_reg, label='train reg')
plt.plot(loss_test_reg, label='test reg')
plt.legend()
plt.show()