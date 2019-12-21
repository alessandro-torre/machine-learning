import numpy as np
from sklearn.preprocessing import StandardScaler
try:
  import matplotlib.pyplot as plt
except ImportError:
  import matplotlib
  matplotlib.use('qt5agg')
  import matplotlib.pyplot as plt



class LinearSVC:
  def __init__(self, C=1.0, normalize=True):
    assert C >= 0, "C must be non-negative."
    self.C = C
    self.normalize = normalize
    self._built = False
    self._fitted = False
    self.w = None
    self.b = None
    self.scaler = StandardScaler()
    self._encode_class = dict()  # to convert classes to signs
    self._decode_class = dict()  # to convert signs to classes
    self.support_ = None  # data points that violate the margins
    self.support_vectors_ = None

  def _build(self, x_shape):
    self.w = np.random.randn(x_shape[1])
    self.b = 0
    self._built = True

  def fit(self, X, Y, learning_rate=1e-5, epochs=1000):
    assert len(X) == len(Y), "Length of x and y do not match"

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

    # Initalize weights
    if not self._built: self._build(x.shape)

    # Gradient descent
    losses = []
    for epoch in range(epochs):
      losses.append(np.mean(self._loss_function(x, y)))
      self.w += - learning_rate * self._dloss_dw(x, y)
      self.b += - learning_rate * self._dloss_db(x, y)

    self._fitted = True
    self.support_ = np.where(y * self._decision_function(x) <= 1)[0]
    self.support_vectors_ = X[self.support_]
    return losses

  def _decision_function(self, x):
    return x.dot(self.w) + self.b

  def _loss_function(self, x, y):
    distance = 0.5 * self.w * self.w
    penalty = np.sum(np.maximum(0, 1 - y * self._decision_function(x)))
    return distance + self.C * penalty

  def _dloss_dw(self, x, y):
    ddistance_dw = self.w
    penalty_idx = np.where(y * self._decision_function(x) < 1)[0]
    dpenalty_dw = - y[penalty_idx].dot(x[penalty_idx])
    return ddistance_dw + self.C * dpenalty_dw

  def _dloss_db(self, x, y):
    ddistance_db = 0
    penalty_idx = np.where(y * self._decision_function(x) < 1)[0]
    dpenalty_db = - y[penalty_idx].sum()
    return ddistance_db + self.C * dpenalty_db

  def predict(self, X):
    assert self._fitted, "This SVC instance is not fitted yet. " + \
      "Call 'fit' with appropriate arguments before using this method."
    if self.normalize:
      self.scaler.partial_fit(X)
      x = self.scaler.transform(X)
    else:
      x = X

    y = np.sign(self._decision_function(x))
    return np.array(list(map(self._decode_class.get, y)))

  def score(self, X, Y):
    Y_hat = self.predict(X)
    return np.mean(Y == Y_hat)

  def plot_decision_boundary(self, X, Y, resolution=100, colors=('b', 'k', 'r')):
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
    grid = [[self._decision_function(np.array([[xr, yr]])) for yr in y_range] for xr in x_range]
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

    plt.show()
