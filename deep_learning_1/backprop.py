import numpy as np
from matplotlib import pyplot as plt

def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
	Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
	O = softmax(Z.dot(W2) - b2)
	return Z, O

def predict(O):
	idx = np.argmax(O, axis=1)
	#one-hot encoding for O
	#O = np.zeros((N,K))
	#for i in xrange(N):
	#	O[i, idx[i]] = 1
	#return O
	return idx

def accuracy(O, Y):
	return np.mean(predict(O) == Y)

def cost(O, T):
	return (T * np.log(O)).sum()

def dcost_dW2(O, T, Z):
	return Z.T.dot(O - T)

def dcost_db2(O, T):
	return (O - T).sum(axis=0) #sum over n

def dcost_dW1(O, T, Z, W2, X):
	return X.T.dot((O - T).dot(W2.T) * Z * (1 - Z))

def dcost_db1(O, T, Z, W2):
	return ((O - T).dot(W2.T) * Z * (1 - Z)).sum(axis=0) #sum over n

def main():
	#create data
	Nclass = 500 #input size per class
	D = 2 # input features
	K = 3 # number of classes
	N = Nclass * K #input size
	X1 = np.random.randn(Nclass,D) + np.array([0, -2])
	X2 = np.random.randn(Nclass,D) + np.array([2, 2])
	X3 = np.random.randn(Nclass,D) + np.array([-2, 2])
	X = np.vstack([X1, X2, X3])
	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
	#one-hot encoding for T
	T = np.zeros((N,K))
	for i in xrange(N):
		T[i, Y[i]] = 1

	#initialise ann
	M = 3 #hidden layer size
	W1 = np.random.randn(D,M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M,K)
	b2 = np.random.randn(K)

	#backpropagation
	learning_rate = 10e-5
	epochs = 10000
	Nprints = 100
	history = np.zeros((epochs, 2))
	for epoch in xrange(epochs):
		Z, O = forward(X, W1, b1, W2, b2)
		W1 -= learning_rate * dcost_dW1(O, T, Z, W2, X)
		b1 -= learning_rate * dcost_db1(O, T, Z, W2)
		W2 -= learning_rate * dcost_dW2(O, T, Z)
		b2 -= learning_rate * dcost_db2(O, T)
		c = np.round(cost(O, T), 2)
		a = np.round(accuracy(O, Y), 3)
		history[epoch] = [c, a]
		if not(epoch % int(float(epochs)/Nprints)):
			print "cost:", c, "accuracy:", a
	plt.plot(history[:,0])
	plt.show()

if __name__ == '__main__':
	main()