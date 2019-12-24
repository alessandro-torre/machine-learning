import numpy as np

class Kernel:

  def __init__(self, kernel, **kwargs):
    kernels = {
      'linear': [self._linear, {'c': 0}],
      'polynomial': [self._polynomial, {'c': 0, 'degree': 2}],
      'sigmoid': [self._sigmoid, {'c': 1, 'gamma': 0.05}],
      'rbf': [self._rbf, {'gamma': None}]
      }

    try:
      self.kernel = kernels[kernel][0]
    except KeyError:
      error_msg = kernel + ' is not a valid kernel. Valid kernels: ' \
                + ', '.join(kernels.keys()) + "."
      raise KeyError(error_msg)

    params = kernels[kernel][1]
    for k, v in kwargs.items():
      error_msg = k + ' is not a valid parameter for ' + kernel + ' kernel. ' \
                + 'Valid parameters: ' + ', '.join(params) + "."
      assert k in params.keys(), error_msg
      params[k] = v
    self._params = params
    self.name = kernel

  def __call__(self, x1, x2):
    return self.kernel(x1, x2, **self._params)

  def _linear(self, x1, x2, c):
    return x1.dot(x2.T) + c

  def _polynomial(self, x1, x2, c, degree):
    return (x1.dot(x2.T) + c) ** degree

  def _sigmoid(self, x1, x2, gamma, c):
    return np.tanh(gamma * x1.dot(x2.T) + c)

  def _rbf(self, x1, x2, gamma):
    if np.ndim(x1) == 1: x1 = x1[np.newaxis, :]
    if np.ndim(x2) == 1: x2 = x2[np.newaxis, :]
    if gamma is None: gamma = 1 / x1.shape[1]
    return np.exp(-gamma * np.linalg.norm(x1[:, np.newaxis] - x2, axis=2))

  def get_params(self):
    return self._params
