import numpy as np
from lib._ann_utils import HIDDEN_ACTIVATIONS
from lib._ann_utils import OUTPUT_ACTIVATIONS
from lib._ann_utils import HIDDEN_DERIVATIVES
from lib._ann_utils import LOSS_FUNCTIONS
from lib._ann_utils import REGULARIZATIONS
from lib._ann_utils import OUTPUT_PREDICTIONS

class ann:
	'''
	ANN with one hidden layer.
	task={'classification', 'regression'}.
	Possible hidden activation functions: activation={'identity', 'sigmoid', 'tanh', 'relu'}.
	Cost function and output activation for classification: cross_entropy with softmax.
	Cost function and output activation for regression: squared_loss with identity.
	The hidden layer can also have dimension zero (i.e. output layer only)

	TODO: logistic output layer for binary classification when K=2

	TODO: sgd (batching)
	TODO: gradient descent with momentum
	TODO: better initialisations of weights (scaling down by n_params)
	TODO: dynamic learning rate
	'''
	def __init__(self, n_features=1, classification_set=None, hidden_layers_shape=5, activation='sigmoid'):
		assert activation in HIDDEN_ACTIVATIONS
		self.activation = activation
		assert isinstance(n_features, int) and n_features > 0
		self.n_features = n_features
		self.classification_set = classification_set
		if classification_set is None:
			self.task = 'regression'
			self.n_outputs = 1
			self.cls2idx = None
			self.idx2cls = None
		else:
			self.task = 'classification'
			assert isinstance(classification_set, set)
			self.n_outputs = len(classification_set)
			# Define dictionaries to one-hot encode and decode classes
			self.cls2idx = dict(zip(classification_set, range(self.n_outputs))) # {[cls:i]}
			self.idx2cls = dict(enumerate(classification_set)) # {[i:cls]}
		if hidden_layers_shape == 0:
			hidden_layers_shape = list()
		elif isinstance(hidden_layers_shape, int):
			assert hidden_layers_shape > 0
			hidden_layers_shape = [hidden_layers_shape]
		else:
			assert isinstance(hidden_layers_shape, list)
			assert all(list(map(lambda x: isinstance(x, int), hidden_layers_shape)))
			assert all(list(map(lambda x: x > 0, hidden_layers_shape)))
		self.layers_shape = [self.n_features] + hidden_layers_shape + [self.n_outputs]
		self.n_layers = len(hidden_layers_shape) + 2
		self.n_param = sum([(self.layers_shape[i]+1)*self.layers_shape[i+1] for i in range(self.n_layers - 1)])
		# Initialise the parameters
		self.W = [];
		self.b = [];
		for i in range(self.n_layers - 1):
			self.W.append(np.random.randn(self.layers_shape[i], self.layers_shape[i + 1])) #/ sqrt(self.n_param))
			self.b.append(np.random.randn(self.layers_shape[i + 1])) #/ sqrt(self.n_param))

	def _onehot(self, Y):
		''' One-hot encoding of target Y based on the classes. '''
		if Y is None:
			return None
		else:
			assert set(Y).issubset(self.classification_set)
			N = len(Y)
			T = np.zeros((N, self.n_outputs))
			T[range(N), map(self.cls2idx.get, Y)] = 1
			return T

	def _forward(self, Z):
		'''
		Feed forward from the input to the output layer.

		Parameters
		__________
		Z : list, length = 1
			The list of activations, initialised with the input layer.

		Returns
		_______
		list, length = self.n_layers
			The list of activations, filled in with the hidden and output activations.
		'''
		assert(Z[0].shape[1] == self.W[0].shape[0]) # ensure that we can .dot()
		for i in range(self.n_layers - 2):
			Z.append(HIDDEN_ACTIVATIONS[self.activation](Z[i].dot(self.W[i]) + self.b[i])) # append Z[i + 1]
		Z.append(OUTPUT_ACTIVATIONS[self.task](Z[-1].dot(self.W[-1]) + self.b[-1])) # append Z[self.n_hidden_layers + 1]
		return Z

	def predict(self, X):
		# Feed forward from X to the output layer
		Z = [X] 
		Z = self._forward(Z)
		# Output layer
		out = Z[-1]
		if out.shape[1] == 1:
			out = out.ravel()
		# In case of classification, out is a pT and needs to be translated to Yhat
		pred = OUTPUT_PREDICTIONS[self.task](out, self.idx2cls)
		return pred

	def loss(self, X, Y):
		''' Calculate the loss function on given X, Y. '''
		if self.task == 'classification':
			assert set(Y).issubset(self.classification_set)
			T = self._onehot(Y)
		else:
			T = Y.reshape(-1, 1)
		return LOSS_FUNCTIONS[self.task](X, T)

	def dloss_dAout(self, T, out):
		'''
		Derivative of the loss function w.r.t the activation of the output layer.
		It consist of the chain rule of the derivative of the loss function w.r.t to the output activation function,
		and the derivative of the activation function w.r.t to its argument.
		Result valid for the following combinations of (loss function & output activation function):
			(cross entropy & softmax), (binary cross entropy & sigmoid), (squared loss & identity).
		Notation: Aout = Z[-2].dot(self.W[-2]) + self.b[-2], i.e. the argument of the output activation function.
		'''
		return (out - T) / T.shape[0] #normalisation over N, coming from mean() in the loss function

	def dloss_dAhid(self, dloss_dA, W, Z):
		'''
		Derivatives for the hidden layer.
		Notation: dloss_dA, W, Z are the weights corresponding to the previous step in backprop.
		If: Ahid = A[i] = Z[i].dot(self.W[i]) + self.b[i]
		Then: input is dloss_dA[i+1], W[i+1], Z[i+1]
		'''
		return dloss_dA.dot(W.T) * HIDDEN_DERIVATIVES[self.activation](Z)
	
	def dloss_dW(self, dloss_dA, Z):
		return Z.T.dot(dloss_dA)
	def dloss_db(self, dloss_dA):
		return dloss_dA.sum(axis=0) #sum over N

	def fit(self, X, Y, Xtest=None, Ytest=None, learning_rate=0.1, reg_rate=0., epochs=5000, verbose=False, n_prints=100,
			adaptive_rate=False, adaptive_coeff=1.5, n_plateau=100, min_mrstd=10e-5, min_lr=10e-10):
		''' Train the neural network with X and Y. '''
		#check for compatibility of input, output, and classification_set
		assert hasattr(X, 'shape') and hasattr(Y, 'shape')
		assert X.shape[0] == Y.shape[0]
		if Xtest is not None and Ytest is not None:
			assert hasattr(X, 'shape') and hasattr(Y, 'shape')
			assert Xtest.shape[1] == X.shape[1]
			assert X.shape[0] == Y.shape[0]
		else:
			Xtest = None
			Ytest = None
			Ttest = None
		assert self.n_features == X.shape[1]
		if self.task == 'classification':
			# For classification, target T is one-hot encoding of Y
			T = self._onehot(Y)
			if Ytest is not None:
				Ttest = self._onehot(Ytest)
		else:
			# For regression, Y.shape = (N, ). We reshape it to (N, 1) to prevent dimensionality issues in dloss_dAout.
			T = Y.reshape(-1, 1)
			if Ytest is not None:
				Ttest = Ytest.reshape(-1, 1)
		# Type checks
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
		# Start training and save history
		history_loss = np.empty(epochs)
		history_acc = np.zeros(epochs)
		history_loss_test = np.empty(epochs)
		history_acc_test = np.zeros(epochs)
		# Feed forward from X to the output layer
		Z = self._forward([X])
		# Print message
		msg = "Training ANN with "
		if self.n_layers == 2:
			msg += "no hidden layer "
		else:
			msg += "shape " + str(self.layers_shape) + " and " + self.activation + " activation "
		msg += "(" + str(self.n_param) + " parameters)..."
		print(msg)
		# Back propagation
		for epoch in range(epochs):
			if learning_rate < min_lr:
				break
			loss_reg = 0
			# Update weights following the gradient descent w.r.t loss function
			for i in range(self.n_layers - 2, -1, -1):
				# Regularisation terms in the loss function.
				loss_reg += reg_rate * REGULARIZATIONS[self.task](self.W[i], self.n_param, deriv=False)
				loss_reg += reg_rate * REGULARIZATIONS[self.task](self.b[i], self.n_param, deriv=False)
				if i == self.n_layers - 2:
					dloss_dA = self.dloss_dAout(T, Z[i + 1]) # here Z[i + 1] = Z[-1], i.e. the output layer
				else:
					dloss_dA = self.dloss_dAhid(dloss_dA, self.W[i + 1], Z[i + 1]) # parameter dloss_dA from the previous step in the for loop
				self.W[i] -= learning_rate * (self.dloss_dW(dloss_dA, Z[i]) + reg_rate * REGULARIZATIONS[self.task](self.W[i], self.n_param))
				self.b[i] -= learning_rate * (self.dloss_db(dloss_dA) + reg_rate * REGULARIZATIONS[self.task](self.b[i], self.n_param))
			# Feed forward with the updated weights
			Z = self._forward([X])
			history_loss[epoch] = LOSS_FUNCTIONS[self.task](Z[-1], T) #+ loss_reg
			history_acc[epoch] = np.mean(OUTPUT_PREDICTIONS[self.task](Z[-1], self.idx2cls) == Y)
			if Xtest is not None:
				Ztest = self._forward([Xtest])
				history_loss_test[epoch] = LOSS_FUNCTIONS[self.task](Ztest[-1], Ttest) #+ loss_reg
				history_acc_test[epoch] = np.mean(OUTPUT_PREDICTIONS[self.task](Ztest[-1], self.idx2cls) == Ytest)
			#halve learning_rate if costs are plateuing (as measured by mrstd = moving relative std)
			mrstd = None
			if epoch > n_plateau:
				last_losses = history_loss[epoch-n_plateau:epoch]
				mrstd = np.std(last_losses) / np.mean(last_losses)
				if adaptive_rate and epoch > epochs/2. and mrstd < min_mrstd:
					learning_rate /= adaptive_coeff
					#print epoch, "learning rate halved to:", learning_rate
			if verbose and (epoch == 0 or (epoch+1) % int(float(epochs) / n_prints) == 0):
				print(epoch+1, " loss:", history_loss[epoch], "mrstd:", mrstd, "acc:", history_acc[epoch], "lr:", learning_rate)
		return history_loss, history_acc, history_loss_test, history_acc_test
