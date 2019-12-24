import numpy as np
from .svm_kernels import Kernel
from sklearn.preprocessing import StandardScaler
try:
  import matplotlib.pyplot as plt
except ImportError:
  import matplotlib
  matplotlib.use('qt5agg')
  import matplotlib.pyplot as plt



class AbstractSVC:
  def __init__(self, C=1.0, normalize=True):
    assert C >= 0, "C must be non-negative."
    self.C = C
    self.normalize = normalize
    self._built = False
    self._fitted = False
    self.scaler = StandardScaler()
    self._encode_class = dict()  # to convert classes to signs
    self._decode_class = dict()  # to convert signs to classes
    self.support_ = None  # data points that violate the margins
    self.support_vectors_ = None

  def preprocess(self, X, Y):
    '''
      Method to normalize features vector X, and to encode classes to [-1, +1].
    '''
    assert len(X) == len(Y), "Length of x and y do not match."

    # Encode classes to {-1, +1}
    cls = set(Y)
    assert len(cls) == 2, "This SVC can only be used for binary classification."
    self._encode_class = dict(zip(cls, [-1, 1]))
    self._decode_class = dict(zip([-1, 1], cls))
    y = np.array(list(map(self._encode_class.get, Y)))

    # Normalize features
    if self.normalize:
      self.scaler.partial_fit(X)
      x = self.scaler.transform(X)
    else:
      x = X

    return x, y

  def _build(self, x_shape):
    ''' Method to initialize weights. '''
    raise NotImplementedError

  def fit(self, X, Y, learning_rate=1e-5, momentum=0.99, epochs=1000):
    '''
      Method to fit the model on the training set.
    '''
    x, y = self.preprocess(X, Y)
    if not self._built:
      self._build(x.shape)
      self._built = True

    # Gradient ascent / descent
    assert learning_rate > 0, "Learning rate must be positive."
    assert momentum >= 0, "Momentum must be non-negative."
    assert epochs > 0, "Epochs must be positive."
    losses = []
    for epoch in range(epochs):
      done = (epoch + 1 == epochs)
      losses.append(self._backprop(x, y, learning_rate, momentum, done))

    self._fitted = True
    self.support_ = np.where(y * self._forward_step(x) <= 1)[0]
    self.support_vectors_ = X[self.support_]
    return losses

  def _backprop(self, x, y, learning_rate, momentum, done):
    ''' Do one step of gradient backpropagation to update model weights. '''
    raise NotImplementedError

  def _forward_step(self, x):
    ''' Do one forward step. '''
    raise NotImplementedError

  def predict(self, X):
    assert self._fitted, "This SVC instance is not fitted yet. " + \
      "Call 'fit' with appropriate arguments before using this method."
    if self.normalize:
      self.scaler.partial_fit(X)
      x = self.scaler.transform(X)
    else:
      x = X
    y = np.sign(self._forward_step(x))
    return np.array(list(map(self._decode_class.get, y)))

  def score(self, X, Y):
    Y_hat = self.predict(X)
    return np.mean(Y == Y_hat)

  def plot_decision_boundary(self, X, Y, resolution=100, colors=('b', 'k', 'r'),
                            title=None):
    assert X.shape[1] == 2, "Method available for 2D input data only."

    # Normalize X and encode Y
    if self.normalize: X = self.scaler.transform(X)
    Y = np.array(list(map(self._encode_class.get, Y)))

    np.warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(X[:,0].min(), X[:,0].max(), resolution)
    y_range = np.linspace(X[:,1].min(), X[:,1].max(), resolution)
    grid = [[self._forward_step(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and make a scatter plot of training data
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
               linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(X[:,0], X[:,1], c=Y, lw=0, alpha=0.3, cmap='seismic')

    # Plot support vectors (non-zero alphas) as circled points (linewidth > 0)
    mask = self.support_
    ax.scatter(X[:,0][mask], X[:,1][mask], c=Y[mask], lw=0.1, cmap='seismic')

    # debug
    ax.scatter([0], [0], c='black', marker='x')

    # debug
    # x_axis = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    # w = model.w
    # b = model.b
    # # w[0]*x + w[1]*y + b = 0
    # y_axis = -(w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, y_axis, color='purple')
    # margin_p = (1 - w[0]*x_axis - b)/w[1]
    # plt.plot(x_axis, margin_p, color='orange')
    # margin_n = -(1 + w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, margin_n, color='orange')

    if title is None: title = self.name
    ax.set_title(title)
    plt.show()



class LinearSVC(AbstractSVC):
  '''
    Linear SVC solving the primal optimization problem via gradient descent.
  '''
  def _build(self, x_shape):
    ''' Initialize weights and momentum. '''
    self.w = np.random.randn(x_shape[-1])
    self.b = np.array(0.)
    self._v_w = np.zeros(x_shape[-1])
    self._v_b = np.array(0.)

  def _backprop(self, x, y, learning_rate, momentum, done):
    self._v_w = momentum * self._v_w + self.__dloss_dw(x, y)
    self._v_b = momentum * self._v_b + self.__dloss_db(x, y)
    self.w -= learning_rate * self._v_w
    self.b -= learning_rate * self._v_b
    return np.mean(self.__loss_function(x, y))

  def __loss_function(self, x, y):
    distance = 0.5 * self.w * self.w
    penalty = np.sum(np.maximum(0, 1 - y * self._forward_step(x)))
    return distance + self.C * penalty

  def __dloss_dw(self, x, y):
    ddistance_dw = self.w
    penalty_idx = np.where(y * self._forward_step(x) < 1)[0]
    dpenalty_dw = - y[penalty_idx].dot(x[penalty_idx])
    return ddistance_dw + self.C * dpenalty_dw

  def __dloss_db(self, x, y):
    ddistance_db = 0
    penalty_idx = np.where(y * self._forward_step(x) < 1)[0]
    dpenalty_db = - y[penalty_idx].sum()
    return ddistance_db + self.C * dpenalty_db

  def _forward_step(self, x):
    return x.dot(self.w) + self.b



class SVC(AbstractSVC):
  '''
    SVC solving the dual optimization problem via gradient descent.
  '''
  def __init__(self, kernel=Kernel('rbf'), C=1.0, normalize=True):
    super().__init__(C, normalize)
    if isinstance(kernel, Kernel):
      self.kernel = kernel
    else:
      self.kernel = Kernel(kernel)
    self.name = self.kernel.name
    self._x = None
    self._y = None
    self._yyK = None

  def _build(self, x_shape):
    ''' Initialize weights and store x, y for kernel calculations. '''
    self._n = x_shape[0]
    self.alpha = self.C * np.random.random(self._n)
    self.b = np.array(0.)
    self._v_alpha = np.zeros(self._n)

  def _backprop(self, x, y, learning_rate, momentum, done):
    if self._yyK is None:
      self._x = x
      self._y = y
      self._yyK = np.outer(y, y) * self.kernel(x, x)
    # Do one step of gradient descent on alpha
    g_alpha = -np.ones(self._n) + self._yyK.dot(self.alpha)
    self._v_alpha = momentum * self._v_alpha + learning_rate * g_alpha
    self.alpha -= self._v_alpha
    # Impose constraint on alpha
    self.alpha = np.minimum(self.C, np.maximum(0, self.alpha))
    # Update b
    if done:
      margin = np.where((self.alpha > 0) & (self.alpha < self.C))[0]
      self.b = np.mean(self._y[margin] - self._forward_step(self._x[margin]))
    # Return loss
    return -np.sum(self.alpha) + 0.5 * self.alpha.dot(self._yyK).dot(self.alpha)

  def _forward_step(self, x):
    return (self.alpha * self._y).dot(self.kernel(self._x, x)) + self.b
