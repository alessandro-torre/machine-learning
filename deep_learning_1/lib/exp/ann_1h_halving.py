# Here we halve the learning rate and we don't update the weights if the cost increases
# Not so smart, as we often get trapped in local minima that are too large
# The issue is that we are not distinguishing between cases where we are happy
# to move around the minima (e.g. we have a good accuracy already and we hope to make it better)
# and when it would be better to jump away (e.g. very large local minima of the cost function)

import numpy as np

class ann_1h:
	def __init__(self, size_i, size_h, class_set):
		assert isinstance(size_i, int) and size_i > 0
		assert isinstance(size_h, int) and size_h > 0
		assert isinstance(class_set, set)
		self.D = size_i
		self.M = size_h
		self.K = len(class_set)
		#store the class set and assign each class to an integer
		self.class_set = class_set
		self.class_enc = dict(zip(class_set, xrange(self.K))) # {[class:i]}
		self.class_dec = dict(enumerate(class_set)) # {[i:class]}
		#initialise the weights
		self.W1 = np.random.randn(self.D, self.M)
		self.b1 = np.random.randn(self.M)
		self.W2 = np.random.randn(self.M, self.K)
		self.b2 = np.random.randn(self.K)

	@staticmethod
	def softmax(A):
		expA = np.exp(A)
		return expA / expA.sum(axis=1, keepdims=True)

	@staticmethod
	def accuracy(Yhat, Y):
		return np.mean(Yhat == Y)

	@staticmethod
	def cost(pT, T):
		return -(T * np.log(pT)).sum()

	def onehot(self, Y):
		assert set(Y).issubset(self.class_set)
		N = len(Y)
		T = np.zeros((N, self.K))
		for i in xrange(N):
			T[i, self.class_enc.get(Y[i])] = 1
		return T

	def forward(self, X, W1=None, b1=None, W2=None, b2=None):
		if W1 is None:
			W1 = self.W1
		if b1 is None:
			b1 = self.b1
		if W2 is None:
			W2 = self.W2
		if b2 is None:
			b2 = self.b2
		Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
		 #pT has dimension of T, i.e. based on one-hot encoding
		pT = self.softmax(Z.dot(W2) + b2)
		return Z, pT

	def pT_to_Yhat(self, pT):
		keys = np.argmax(pT, axis=1)
		return map(self.class_dec.get, keys)

	def predict(self, X):
		Z, pT = self.forward(X)
		return self.pT_to_Yhat(pT)

	def dcost_dW2(self, pT, T, Z):
		return Z.T.dot(pT - T)

	def dcost_db2(self, pT, T):
		return (pT - T).sum(axis=0) #sum over n

	def dcost_dW1(self, pT, T, Z, X):
		return X.T.dot((pT - T).dot(self.W2.T) * Z * (1 - Z))

	def dcost_db1(self, pT, T, Z):
		return ((pT - T).dot(self.W2.T) * Z * (1 - Z)).sum(axis=0) #sum over n

	def fit(self, X, Y, learning_rate=10e-5, epochs=10000, verbose=True, n_prints=100):
		#check for compatibility of input, output, and class_set
		assert X.shape[0] == Y.shape[0]
		assert self.D == X.shape[1]
		assert set(Y).issubset(self.class_set)
		#sanity checks
		assert learning_rate > 0
		assert isinstance(epochs, int) and epochs > 0
		assert isinstance(n_prints, int) and n_prints > 0
		#T is target (one-hot encoding of Y)
		T = self.onehot(Y)
		#initial cost and accuracy
		Z, pT = self.forward(X)
		c = self.cost(pT, T)
		a = self.accuracy(self.pT_to_Yhat(pT), Y)
		#start training and save history
		history = np.zeros((epochs, 2))
		for epoch in xrange(epochs):
			#gradient descent deltas
			dW2 = -learning_rate * self.dcost_dW2(pT, T, Z)
			db2 = -learning_rate * self.dcost_db2(pT, T)
			dW1 = -learning_rate * self.dcost_dW1(pT, T, Z, X)
			db1 = -learning_rate * self.dcost_db1(pT, T, Z)
			#check result before updating weights
			Z, pT = self.forward(X, self.W1+dW1, self.b1+db1, self.W2+dW2, self.b2+db2)
			c_new = self.cost(pT, T)
			a_new = self.accuracy(self.pT_to_Yhat(pT), Y)
			#update weights only if cost is lower
			if c_new <= c:
				c = c_new
				a = a_new
				self.W2 += dW2
				self.b2 += db2
				self.W1 += dW1
				self.b1 += db1
				if not(epoch % int(float(epochs)/n_prints)):
					print "cost:", c, "accuracy:", a
			#otherwise halve learning_rate
			else:
				learning_rate /= 2;
				print "cost:", c, "new cost:", c_new, "learning rate halved to:", learning_rate
			#save history
			history[epoch] = [c, a]
		return history
