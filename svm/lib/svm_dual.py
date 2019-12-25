import numpy as np
from .svm_base import AbstractSVC
from .svm_kernels import Kernel



class SVC(AbstractSVC):
  '''
    SVC implementation based on the dual lagrangian.
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
