# ANN with one hidden layer with logistic activation function.
# The hidden layer can also have dimension zero (i.e. softmax output layer only)
# TODO: add validation set

import numpy as np

class ann_1h:
	def __init__(self, size_i, size_h, cls_set):
		assert isinstance(size_i, int) and size_i > 0
		assert isinstance(size_h, int) and size_h >= 0 #allow for no hidden layer
		assert isinstance(cls_set, set)
		self.D = size_i
		self.M = size_h
		self.K = len(cls_set)
		#store the class set and assign each class to an integer
		self.cls_set = cls_set
		self.cls2idx = dict(zip(cls_set, xrange(self.K))) # {[cls:i]}
		self.idx2cls = dict(enumerate(cls_set)) # {[i:cls]}
		#initialise the weights
		if self.M > 0:
			self.W1 = np.random.randn(self.D, self.M)
			self.b1 = np.random.randn(self.M)
			self.W2 = np.random.randn(self.M, self.K)
			self.b2 = np.random.randn(self.K)
		else:
			self.W2 = np.random.randn(self.D, self.K)
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
		return -(T * np.log(pT)).sum() #cross entropy

	def onehot(self, Y):
		assert set(Y).issubset(self.cls_set)
		N = len(Y)
		T = np.zeros((N, self.K))
		#for i in xrange(N):
		#	T[i, self.cls2idx.get(Y[i])] = 1
		T[xrange(N), map(self.cls2idx.get, Y)] = 1
		return T

	def forward(self, X):
		if self.M > 0:
			Z = 1 / (1 + np.exp(-X.dot(self.W1) - self.b1)) #logistic activation function
		else:
			Z = X
		pT = self.softmax(Z.dot(self.W2) + self.b2) #pT has dimension of T, i.e. based on one-hot encoding
		return Z, pT

	def pT_to_Yhat(self, pT):
		keys = np.argmax(pT, axis=1)
		return map(self.idx2cls.get, keys)

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

	def fit(self, X, Y, learning_rate=10e-5, epochs=10000, verbose=False, n_prints=100, adaptive_rate=False, adaptive_coeff=1.5, n_plateau=100, min_mrstd=10e-5, min_lr=10e-10):
		#check for compatibility of input, output, and cls_set
		assert X.shape[0] == Y.shape[0]
		assert self.D == X.shape[1]
		assert set(Y).issubset(self.cls_set)
		#sanity checks
		assert isinstance(learning_rate, float) and learning_rate > 0
		assert isinstance(epochs, int) and epochs > 0
		assert isinstance(verbose, bool)
		assert isinstance(n_prints, int) and n_prints > 0
		assert isinstance(adaptive_rate, bool)
		assert isinstance(adaptive_coeff, float) and adaptive_coeff > 0
		assert isinstance(n_plateau, int) and n_plateau > 0
		assert isinstance(min_mrstd, float) and min_mrstd > 0
		assert isinstance(min_lr, float) and min_lr > 0
		#T is target (one-hot encoding of Y)
		T = self.onehot(Y)
		#start training and save history
		history_c = np.zeros((epochs))
		history_a = np.zeros((epochs))
		Z, pT = self.forward(X)
		print "Training ANN with", self.M, "hidden neurons..."
		for epoch in xrange(epochs):
			if learning_rate < min_lr:
				break
			#update weights
			self.W2 -= learning_rate * self.dcost_dW2(pT, T, Z)
			self.b2 -= learning_rate * self.dcost_db2(pT, T)
			if self.M > 0:
				self.W1 -= learning_rate * self.dcost_dW1(pT, T, Z, X)
				self.b1 -= learning_rate * self.dcost_db1(pT, T, Z)
			#feed-forward
			Z, pT = self.forward(X)
			history_c[epoch] = self.cost(pT, T)
			history_a[epoch] = self.accuracy(self.pT_to_Yhat(pT), Y)
			#halve learning_rate if costs are plateuing (as measured by mrstd = moving relative std)
			if epoch < n_plateau:
				mrstd = None
			else:
				last_c = history_c[epoch-n_plateau:epoch]
				mrstd = np.std(last_c) / np.mean(last_c)
				if adaptive_rate and epoch > epochs/2. and mrstd < min_mrstd:
					learning_rate /= adaptive_coeff
					#print epoch, "learning rate halved to:", learning_rate
			if verbose and (epoch == 0 or (epoch+1) % int(float(epochs) / n_prints) == 0):
				print epoch+1, " c:", history_c[epoch], "mrstd:", mrstd, "a:", history_a[epoch], "lr:", learning_rate
		return history_c, history_a
