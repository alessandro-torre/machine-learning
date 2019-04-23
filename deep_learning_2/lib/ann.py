import numpy as np
from sklearn.utils import shuffle
from lib._ann_utils import HIDDEN_ACTIVATIONS
from lib._ann_utils import OUTPUT_ACTIVATIONS
from lib._ann_utils import HIDDEN_DERIVATIVES
from lib._ann_utils import LOSS_FUNCTIONS
from lib._ann_utils import REGULARIZATIONS
from lib._ann_utils import OUTPUT_PREDICTIONS


class ann:
    """
    ANN with one hidden layer.
    task={'classification', 'regression'}.
    Possible hidden activation functions: activation={'identity', 'sigmoid', 'tanh', 'relu'}.
    Cost function and output activation for classification: cross_entropy with softmax.
    Cost function and output activation for regression: squared_loss with identity.
    The hidden layer can also have dimension zero (i.e. output layer only)

    TODO: logistic output layer for binary classification when K=2
    TODO: measure training time

    TODO: better initialisations of weights (scaling down by n_params)
    TODO: early stopping
    """

    def __init__(self, n_features=1, classification_set=None, hidden_layers_shape=0, activation='sigmoid'):
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
            self.cls2idx = dict(zip(classification_set, range(self.n_outputs)))  # {[cls:i]}
            self.idx2cls = dict(enumerate(classification_set))  # {[i:cls]}
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
        self.n_param = sum([(self.layers_shape[i] + 1) * self.layers_shape[i + 1] for i in range(self.n_layers - 1)])
        self._init_params()

    def _init_params(self):
        self.W = []
        self.gW = []
        self.dW = []
        self.b = []
        self.gb = []
        self.db = []
        # Adaptive learning rate parameters
        self.cache_gW = []
        self.cache_gb = []
        self.cache_gW2 = []
        self.cache_gb2 = []
        self.gW_adj = []
        self.gb_adj = []
        for i in range(self.n_layers - 1):
            self.W.append(np.random.randn(self.layers_shape[i], self.layers_shape[i + 1]))  # / sqrt(self.n_param))
            self.gW.append(np.zeros((self.layers_shape[i], self.layers_shape[i + 1])))
            self.dW.append(np.zeros((self.layers_shape[i], self.layers_shape[i + 1])))
            self.cache_gW.append(np.ones((self.layers_shape[i], self.layers_shape[i + 1])))
            self.cache_gW2.append(np.ones((self.layers_shape[i], self.layers_shape[i + 1])))
            self.gW_adj.append(np.zeros((self.layers_shape[i], self.layers_shape[i + 1])))
            self.b.append(np.random.randn(self.layers_shape[i + 1]))  # / sqrt(self.n_param))
            self.gb.append(np.zeros(self.layers_shape[i + 1]))
            self.db.append(np.zeros(self.layers_shape[i + 1]))
            self.cache_gb.append(np.ones(self.layers_shape[i + 1]))
            self.cache_gb2.append(np.ones(self.layers_shape[i + 1]))
            self.gb_adj.append(np.zeros(self.layers_shape[i + 1]))

    def _onehot(self, Y):
        """ One-hot encoding of target Y based on the classes. """
        if Y is None:
            return None
        else:
            assert set(Y).issubset(self.classification_set)
            N = len(Y)
            T = np.zeros((N, self.n_outputs))
            T[range(N), map(self.cls2idx.get, Y)] = 1
            return T

    def _forward(self, X):
        """
        Feed forward from the input to the output layer.

        Parameters
        __________
        Z : list, length = 1
            The list of activations, initialised with the input layer.

        Returns
        _______
        list, length = self.n_layers
            The list of activations, filled in with the hidden and output activations.
        """
        assert (X.shape[1] == self.W[0].shape[0])  # ensure that we can .dot()
        Z = [X]
        for i in range(self.n_layers - 2):
            Z.append(HIDDEN_ACTIVATIONS[self.activation](Z[i].dot(self.W[i]) + self.b[i]))  # append Z[i + 1]
        Z.append(
            OUTPUT_ACTIVATIONS[self.task](Z[-1].dot(self.W[-1]) + self.b[-1]))  # append Z[self.n_hidden_layers + 1]
        return Z

    def predict(self, X):
        Z = self._forward(X)
        # Output layer
        out = Z[-1]
        if out.shape[1] == 1:
            out = out.ravel()
        # In case of classification, out is a pT and needs to be translated to Yhat
        pred = OUTPUT_PREDICTIONS[self.task](out, self.idx2cls)
        return pred

    def loss(self, X, Y):
        """ Calculate the loss function on given X, Y. """
        if self.task == 'classification':
            assert set(Y).issubset(self.classification_set)
            T = self._onehot(Y)
        else:
            T = Y.reshape(-1, 1)
        return LOSS_FUNCTIONS[self.task](X, T)

    def dloss_dAout(self, T, out):
        """
        Derivative of the loss function w.r.t the activation of the output layer.
        It consist of the chain rule of the derivative of the loss function w.r.t to the output activation function,
        and the derivative of the activation function w.r.t to its argument.
        Result valid for the following combinations of (loss function & output activation function):
            (cross entropy & softmax), (binary cross entropy & sigmoid), (squared loss & identity).
        Notation: Aout = Z[-2].dot(self.W[-2]) + self.b[-2], i.e. the argument of the output activation function.
        """
        return (out - T) / T.shape[0]  # normalisation over N, coming from mean() in the loss function

    def dloss_dAhid(self, dloss_dA, W, Z):
        """
        Derivatives for the hidden layer.
        Notation: dloss_dA, W, Z are the weights corresponding to the previous step in backprop.
        If: Ahid = A[i] = Z[i].dot(self.W[i]) + self.b[i]
        Then: input is dloss_dA[i+1], W[i+1], Z[i+1]
        """
        return dloss_dA.dot(W.T) * HIDDEN_DERIVATIVES[self.activation](Z)

    def dloss_dW(self, dloss_dA, Z):
        return Z.T.dot(dloss_dA)

    def dloss_db(self, dloss_dA):
        return dloss_dA.sum(axis=0)  # sum over N

    def fit(self, X, Y, Xtest=None, Ytest=None, refit=False,
            adaptive_learning='constant', learning_rate=0.01, beta1=0.90, beta2=0.99, epsilon=0.0001,
            reg_rate=0., epochs=5000, batch_size=None, momentum=0.90, nesterov=True,
            verbose=False, n_prints=100):
        """ Train the neural network with X and Y. """
        # Check for compatibility of input, output, and classification_set
        assert hasattr(X, 'shape') and hasattr(Y, 'shape')
        assert X.shape[0] == Y.shape[0]
        if Xtest is not None and Ytest is not None:
            assert hasattr(X, 'shape') and hasattr(Y, 'shape')
            assert Xtest.shape[1] == X.shape[1]
            assert X.shape[0] == Y.shape[0]
        else:
            Xtest = Ytest = None
        assert self.n_features == X.shape[1]
        if self.task == 'classification':
            # For classification, target T is one-hot encoding of Y
            T = self._onehot(Y)
            Ttest = self._onehot(Ytest) if (Ytest is not None) else None
        else:
            # For regression, Y.shape = (N, ). We reshape it to (N, 1) to prevent dimensionality issues in dloss_dAout.
            T = Y.reshape(-1, 1)
            Ttest = Ytest.reshape(-1, 1) if (Ytest is not None) else None
        # Type checks
        assert isinstance(refit, bool)
        assert adaptive_learning in ['constant', 'adagrad', 'rmsprop', 'adam']
        assert isinstance(learning_rate, float) and learning_rate > 0
        assert isinstance(epsilon, float) and epsilon > 0
        assert isinstance(beta1, float) and 0 <= beta1 < 1
        assert isinstance(beta2, float) and 0 <= beta2 <= 1
        assert isinstance(reg_rate, float) and reg_rate >= 0
        assert isinstance(epochs, int) and epochs > 0
        assert (isinstance(batch_size, int) and batch_size > 0) or (batch_size is None)
        assert isinstance(momentum, float) and momentum >= 0
        assert isinstance(nesterov, bool)
        assert isinstance(verbose, bool)
        assert isinstance(n_prints, int) and n_prints > 0
        # Set learning rate cache weights
        beta1 = beta1 if adaptive_learning == 'adam' else 0
        cache1_w1 = beta1
        cache1_w2 = 1 - beta1
        beta2 = beta2 if adaptive_learning in ['rmsprop', 'adam'] else 1
        cache2_w1 = beta2
        cache2_w2 = 1 - beta2
        if adaptive_learning == 'adagrad':
            cache2_w2 = 1
        # Do not mix Adam and momentum
        if adaptive_learning == 'adam':
            momentum = 0
        # Determine number of batches
        batch_size = min(batch_size, X.shape[0]) if batch_size is not None else X.shape[0]
        n_batches = X.shape[0] // batch_size
        # Start training and save history
        if refit:
            self._init_params()
        history_loss = np.empty(epochs)
        history_acc = np.zeros(epochs)
        history_loss_test = np.empty(epochs)
        history_acc_test = np.zeros(epochs)
        # Print message
        msg = "Training ANN with "
        if self.n_layers == 2:
            msg += "no hidden layer "
        else:
            msg += "shape " + str(self.layers_shape) + " and " + self.activation + " activation "
        msg += "(" + str(self.n_param) + " parameters)..."
        print(msg)
        # Loop of back propagation and feed forward
        for epoch in range(epochs):
            loss_reg = 0
            # Loop over the batches
            Xtrain, Ttrain = shuffle(X, T)
            for batch in range(n_batches):
                # Feed forward from X to the output layer
                x = Xtrain[batch * batch_size: (batch + 1) * batch_size, :]
                t = Ttrain[batch * batch_size: (batch + 1) * batch_size, :]
                z = self._forward(x)
                # Calculate dloss_dA for the output layer
                dloss_dA = self.dloss_dAout(t, z[-1])
                # Update weights following the gradient descent w.r.t loss function, layer by layer
                for i in range(self.n_layers - 2, -1, -1):
                    # Regularisation terms in the loss function
                    loss_reg += reg_rate * REGULARIZATIONS[self.task](self.W[i], self.n_param, deriv=False)
                    loss_reg += reg_rate * REGULARIZATIONS[self.task](self.b[i], self.n_param, deriv=False)
                    if i < self.n_layers - 2:
                        # Parameter dloss_dA (r.h.s.) from the previous (outer) layer
                        dloss_dA = self.dloss_dAhid(dloss_dA, self.W[i + 1], z[i + 1])
                    # Gradients with regularisation term
                    self.gW[i] = self.dloss_dW(dloss_dA, z[i]) + reg_rate * REGULARIZATIONS[self.task](self.W[i], self.n_param)
                    self.gb[i] = self.dloss_db(dloss_dA) + reg_rate * REGULARIZATIONS[self.task](self.b[i], self.n_param)
                    # Caches for adaptive learning (adagrad, rmsprop, adam)
                    self.cache_gW[i] = cache1_w1 * self.cache_gW[i] + cache1_w2 * self.gW[i]
                    self.cache_gb[i] = cache1_w1 * self.cache_gb[i] + cache1_w2 * self.gb[i]
                    self.cache_gW2[i] = cache2_w1 * self.cache_gW2[i] + cache2_w2 * self.gW[i] ** 2
                    self.cache_gb2[i] = cache2_w1 * self.cache_gb2[i] + cache2_w2 * self.gb[i] ** 2
                    # Adaptive gradient with bias correction
                    t = epoch * batch + 1
                    corr1 = (1 - beta1 ** t) if beta1 < 1 else 1
                    corr2 = (1 - beta2 ** t) if beta2 < 1 else 1
                    self.gW_adj[i] = self.cache_gW[i] / np.sqrt(self.cache_gW2[i] + epsilon) * corr2 / corr1
                    self.gb_adj[i] = self.cache_gb[i] / np.sqrt(self.cache_gb2[i] + epsilon) * corr2 / corr1
                    # Gradient with momentum
                    self.dW[i] = momentum * self.dW[i] - self.gW_adj[i]
                    self.db[i] = momentum * self.db[i] - self.gb_adj[i]
                    self.W[i] += learning_rate * (self.dW[i] + nesterov * ((momentum - 1) * self.dW[i] - self.gW_adj[i]))
                    self.b[i] += learning_rate * (self.db[i] + nesterov * ((momentum - 1) * self.db[i] - self.gb_adj[i]))
            # Use the updated weights
            Z = self._forward(X)
            history_loss[epoch] = LOSS_FUNCTIONS[self.task](Z[-1], T)  # + loss_reg
            history_acc[epoch] = np.mean(OUTPUT_PREDICTIONS[self.task](Z[-1], self.idx2cls) == Y)
            if Xtest is not None:
                Ztest = self._forward(Xtest)
                history_loss_test[epoch] = LOSS_FUNCTIONS[self.task](Ztest[-1], Ttest)  # + loss_reg
                history_acc_test[epoch] = np.mean(OUTPUT_PREDICTIONS[self.task](Ztest[-1], self.idx2cls) == Ytest)
            if verbose and (epoch == 0 or (epoch + 1) % int(float(epochs) / n_prints) == 0):
                print(epoch + 1, " loss:", history_loss[epoch], "acc:", history_acc[epoch],
                      "lr:", learning_rate)
        return {'loss': history_loss, 'acc': history_acc, 'loss_v': history_loss_test, 'acc_v': history_acc_test}
