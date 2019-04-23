import numpy as np


# Activation functions for the hidden layers and the output layer.
def identity(A):
	return A
def softmax(A):
	expA = np.exp(A)
	return expA / expA.sum(axis=1, keepdims=True)
def sigmoid(A):
	return 1 / (1 + np.exp(-A))
def tanh(A):
	return np.tanh(A)
def relu(A):
	return A * (A > 0)
# Dictionaries for the hidden and output activation functions.
# For the output layer, the task is the key.
HIDDEN_ACTIVATIONS = {
	'identity': identity,
	'sigmoid': sigmoid,
	'tanh': tanh,
	'relu': relu
}
OUTPUT_ACTIVATIONS = {
	'classification': softmax,
	'regression': identity
}


# Derivatives of the activation functions for the hidden layers.
# Note that Z is already the output of the activation function.
def d_identity(Z):
	return 1
def d_sigmoid(Z):
	return Z * (1 - Z)
def d_tanh(Z):
	return 1 - Z**2
def d_relu(Z):
	return Z > 0
# Dictionary for the derivatives of the hidden activation functions.
HIDDEN_DERIVATIVES = {
	'identity': d_identity,
	'sigmoid': d_sigmoid,
	'tanh': d_tanh,
	'relu': d_relu
}

# Loss functions to be minimized by the neural network.
# cross_entropy for classification (with softmax output activation), squared loss for regression.
# We do not include categorical_cross_entropy for binary classification (with sigmoid output activation),
# as it can be covered by softmax with K=2.
def cross_entropy(out, T):
	pT = out
	return -(T * np.log(pT)).mean()
def squared_loss(out, T):
	Yhat = out
	return 0.5 * ((Yhat - T)**2).mean()
# Dictionary for the loss functions. The task is the key.
LOSS_FUNCTIONS = {
	'classification': cross_entropy,
	'regression': squared_loss
}

# L1 and L2 regularization functions and derivatives
def L1(param, n_param, deriv=True):
	if deriv:
		result = np.sign(param) / n_param
	else:
		result = np.abs(param).sum() / n_param
	return result
def L2(param, n_param, deriv=True):
	if deriv:
		result = param / n_param
	else:
		param = param.ravel()
		result = 0.5 * param.dot(param) / n_param
	return result
# Dictionary for the loss functions. The task is the key.
REGULARIZATIONS = {
	'classification': L2,
	'regression': L1
}

# Final transformation to make predictions from the neural network output.
def Yhat_to_Yhat(out, idx2cls=None):
	Yhat = out
	return Yhat
def pT_to_Yhat(out, idx2cls):
	pT = out
	keys = np.argmax(pT, axis=1)
	Yhat = map(idx2cls.get, keys)
	return Yhat
# Dictionary for doing predictions from the neural network output. The task is the key.
OUTPUT_PREDICTIONS = {
	'classification': pT_to_Yhat,
	'regression': Yhat_to_Yhat
}
