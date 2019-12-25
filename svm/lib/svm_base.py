import numpy as np
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
      Normalize features vector X, and encode classes to {-1, +1}.
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
    ''' Initialize weights. '''
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
