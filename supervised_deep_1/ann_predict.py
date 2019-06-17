import numpy as np
from lib.process import get_data

X, T = get_data()

N, D = X.shape
K = len(set(T))
M = 5 #neurons in the hidden layer

W1 = np.random.randn(D, M)
W2 = np.random.randn(M, K)
b1 = np.zeros(M)
b2 = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2)

def predict(prob):
	return np.argmax(prob, axis=1)

def accuracy(P, T):
	return np.mean(P == T)

pY = forward(X, W1, b1, W2, b2)
Y = predict(pY)
print("accuracy:", accuracy(Y,T))
