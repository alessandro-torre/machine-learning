# ANN with one hidden layer.
# task={'classification', 'regression'}.
# Possible hidden activation functions: activation={'identity', 'sigmoid', 'tanh', 'relu'}.
# Cost function and output activation for classification: cross_entropy with softmax.
# Cost function and output activation for regression: squared_loss with identity.
# The hidden layer can also have dimension zero (i.e. output layer only)
#
# TODO: multi hidden layers
# TODO: validation set
# TODO: logistic output layer for binary classification W_hen K=2

import numpy as np
from _ann_utils import HIDDEN_ACTIVATIONS
from _ann_utils import OUTPUT_ACTIVATIONS
from _ann_utils import HIDDEN_DERIVATIVES
from _ann_utils import LOSS_FUNCTIONS
from _ann_utils import REGULARIZATIONS
from _ann_utils import OUTPUT_PREDICTIONS

class ann_1h:
	def __init__(self, task, size_i, size_h, cls_set=None, activation='sigmoid'):
		assert (task in OUTPUT_ACTIVATIONS.keys())
		assert (activation in HIDDEN_ACTIVATIONS)
		assert isinstance(size_i, int) and size_i > 0
		assert isinstance(size_h, int) and size_h >= 0 #allow for no hidden layer
		if task == 'classification':
			assert isinstance(cls_set, set)
			self.K = len(cls_set)
			# dictionaries to map classes to integers and viceversa
			self.cls2idx = dict(zip(cls_set, xrange(self.K))) # {[cls:i]}
			self.idx2cls = dict(enumerate(cls_set)) # {[i:cls]}
		else: #regression
			self.K = 1
			self.cls2idx = None
			self.idx2cls = None
		self.task = task
		self.D = size_i
		self.M = size_h
		self.cls_set = cls_set
		self.activation = activation
		#initialise the weights
		if self.M > 0:
			self.W_h = np.random.randn(self.D, self.M)
			self.b_h = np.random.randn(self.M)
			self.W_o = np.random.randn(self.M, self.K)
			self.b_o = np.random.randn(self.K)
			self.n_param = self.M * (self.D + 1) + self.K * (self.M + 1)
		else: #no hidden layer
			self.W_o = np.random.randn(self.D, self.K)
			self.b_o = np.random.randn(self.K)
			self.n_param = self.K * (self.D + 1)

	def _onehot(self, Y):
		assert set(Y).issubset(self.cls_set)
		N = len(Y)
		T = np.zeros((N, self.K))
		T[xrange(N), map(self.cls2idx.get, Y)] = 1
		return T

	# out=Yhat for regression, out=pT for classification
	def _forward(self, X):
		if self.M > 0:
			A_h = X.dot(self.W_h) + self.b_h #hidden layer
			Z = HIDDEN_ACTIVATIONS[self.activation](A_h)
		else:
			Z = X #no hidden layer
		A_o = Z.dot(self.W_o) + self.b_o #output layer
		out = OUTPUT_ACTIVATIONS[self.task](A_o)
		return Z, out

	def predict(self, X):
		Z, out = self._forward(X)
		if out.shape[1] == 1:
			out = out.ravel()
		return OUTPUT_PREDICTIONS[self.task](out, self.idx2cls)

	# Derivative for the output layer. Result valid for the following combinations of (loss function & output activation):
	# (cross entropy & softmax), (binary cross entropy & sigmoid), (squared loss & identity).
	# Notation: Ao = Z.dot(self.Wo) + self.bo, i.e. the argument of the output activation function.
	def dloss_dA_o(self, out, T):
		return (out - T) / T.shape[0] #normalisation over N, coming from mean() in the loss function
	# Derivatives for the hidden layer.
	# Notation: Ah = X.dot(self.W_h) + self.b_h, i.e. the argument of the hidden activation function.
	def dloss_dA_h(self, dloss_dA_o, Z):
		return dloss_dA_o.dot(self.W_o.T) * HIDDEN_DERIVATIVES[self.activation](Z)
	# Derivatives for W and b
	def dloss_dW(self, dloss_dA, Z):
		return Z.T.dot(dloss_dA)
	def dloss_db(self, dloss_dA):
		return dloss_dA.sum(axis=0) #sum over N

	# Train the ann with X and Y.
	def fit(self, X, Y, learning_rate=0.01, reg_rate=0., epochs=10000, verbose=False, n_prints=100, adaptive_rate=False, adaptive_coeff=1.5, n_plateau=100, min_mrstd=10e-5, min_lr=10e-10):
		#check for compatibility of input, output, and cls_set
		assert X.shape[0] == Y.shape[0]
		assert self.D == X.shape[1]
		if self.task == 'classification':
			assert set(Y).issubset(self.cls_set)
			# For classification, target T is one-hot encoding of Y
			T = self._onehot(Y)
		else:
			# For regression, Y.shape = (N, ).
			# We reshape it to (N, 1) to prevent dimensionality issues in dloss_dA_o and dloss_dA_h.
			T = Y.reshape(Y.shape[0], 1)
		#sanity checks
		assert isinstance(learning_rate, float) and learning_rate > 0
		assert isinstance(reg_rate, float) and reg_rate >= 0
		assert isinstance(epochs, int) and epochs > 0
		assert isinstance(verbose, bool)
		assert isinstance(n_prints, int) and n_prints > 0
		assert isinstance(adaptive_rate, bool)
		assert isinstance(adaptive_coeff, float) and adaptive_coeff > 0
		assert isinstance(n_plateau, int) and n_plateau > 0
		assert isinstance(min_mrstd, float) and min_mrstd > 0
		assert isinstance(min_lr, float) and min_lr > 0
		#start training and save history
		history_loss = np.zeros((epochs))
		history_acc = np.zeros((epochs))
		Z, out = self._forward(X)
		msg = "Training ANN with"
		if self.M == 0:
			print msg, "no hidden layer..."
		else:
			print msg, self.M, "hidden neurons and", self.activation, "activation..."
		for epoch in xrange(epochs):
			if learning_rate < min_lr:
				break
			# Calculate loss
			loss = (LOSS_FUNCTIONS[self.task](out, T) + 
				reg_rate * REGULARIZATIONS[self.task](self.W_o, self.n_param, deriv=False) +
				reg_rate * REGULARIZATIONS[self.task](self.b_o, self.n_param, deriv=False))
			# Update weights following the gradient descent w.r.t loss function.
			dloss_dA_o = self.dloss_dA_o(out, T)
			self.W_o -= learning_rate * self.dloss_dW(dloss_dA_o, Z) + reg_rate * REGULARIZATIONS[self.task](self.W_o, self.n_param)
			self.b_o -= learning_rate * self.dloss_db(dloss_dA_o) + reg_rate * REGULARIZATIONS[self.task](self.b_o, self.n_param)
			if self.M > 0:
				loss += reg_rate * REGULARIZATIONS[self.task](self.W_h, self.n_param, deriv=False)
				loss += reg_rate * REGULARIZATIONS[self.task](self.b_h, self.n_param, deriv=False)
				dloss_dA_h = self.dloss_dA_h(dloss_dA_o, Z)
				self.W_h -= learning_rate * self.dloss_dW(dloss_dA_h, X) + reg_rate * REGULARIZATIONS[self.task](self.W_h, self.n_param)
				self.b_h -= learning_rate * self.dloss_db(dloss_dA_h) + reg_rate * REGULARIZATIONS[self.task](self.b_h, self.n_param)
			# Feed-forward
			Z, out = self._forward(X)
			# Loss and accuracy history
			history_loss[epoch] =  loss
			if self.task == 'classification':
				history_acc[epoch] = np.mean(OUTPUT_PREDICTIONS[self.task](out, self.idx2cls) == Y)
			else:
				history_acc[epoch] = None
			#halve learning_rate if costs are plateuing (as measured by mrstd = moving relative std)
			mrstd = None
			if epoch > n_plateau:
				last_losses = history_loss[epoch-n_plateau:epoch]
				mrstd = np.std(last_losses) / np.mean(last_losses)
				if adaptive_rate and epoch > epochs/2. and mrstd < min_mrstd:
					learning_rate /= adaptive_coeff
					#print epoch, "learning rate halved to:", learning_rate
			if verbose and (epoch == 0 or (epoch+1) % int(float(epochs) / n_prints) == 0):
				print epoch+1, " loss:", history_loss[epoch], "mrstd:", mrstd, "acc:", history_acc[epoch], "lr:", learning_rate
		return history_loss, history_acc
