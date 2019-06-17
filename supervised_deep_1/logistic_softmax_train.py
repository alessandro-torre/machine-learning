# Idea here: multiclass classification with no hidden layer and a softmax output layer,
# as a natural extension of logistic regression (one neuron).

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from process import get_data

def onehot(Y, K):
	N = len(Y)
	T = np.zeros((N, K))
	T[xrange(N), Y] = 1
	return T

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

N, D = X.shape
K = len(set(Y))
Ntest = 100
Ntrain = N - Ntest

Xtrain = X[:Ntrain]
Ytrain = Y[:Ntrain]
Xtest = X[-Ntest:]
Ytest = Y[-Ntest:]

W = np.random.randn(D, K)
b = np.random.randn(K)

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
	return softmax(X.dot(W) + b)

def pT_to_Yhat(pT):
	return np.argmax(pT, axis=1)

def accuracy(Y, Yhat):
	return np.mean(Y == Yhat)

def cost(T, pT):
	return -np.mean(T * np.log(pT)) #cross entropy

def dcost_dW(T, pT, X):
	return X.T.dot(pT - T)

def dcost_db(T, pT):
	return (pT - T).sum(axis=0)

#define training properties
epochs = 10000
learning_rate = 0.001
n_prints = 10
#start training
Ttrain = onehot(Ytrain, K)
Ttest = onehot(Ytest, K)
pTtrain = forward(Xtrain, W, b)
train_costs = []
test_costs = []
for epoch in xrange(epochs):
	W -= learning_rate * dcost_dW(Ttrain, pTtrain, Xtrain)
	b -= learning_rate * dcost_db(Ttrain, pTtrain)
	pTtrain = forward(Xtrain, W, b)
	pTtest = forward(Xtest, W, b)
	train_costs.append(cost(Ttrain, pTtrain))
	test_costs.append(cost(Ttest, pTtest))
	if epoch == 0 or (epoch+1) % int(float(epochs) / n_prints) == 0:
		print epoch+1, train_costs[epoch], test_costs[epoch]

print "Final train accuracy:", accuracy(Ytrain, pT_to_Yhat(pTtrain))
print "Final test accuracy:", accuracy(Ytest, pT_to_Yhat(pTtest))

legend_train, = plt.plot(train_costs, label=train_costs)
legend_test, = plt.plot(test_costs, label=test_costs)
plt.legend([legend_train, legend_test])
plt.show()
