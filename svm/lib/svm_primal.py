import numpy as np
from .svm_base import AbstractSVC
from .svm_kernels import Kernel



class LinearSVC(AbstractSVC):
  '''
    Linear SVC implementation based on the primal lagrangian.
  '''
  def _build(self, x_shape):
    ''' Initialize weights and momentum. '''
    self.w = np.random.randn(x_shape[-1])
    self.b = np.array(0.)
    self._v_w = np.zeros(x_shape[-1])
    self._v_b = np.array(0.)

  def _backprop(self, x, y, learning_rate, momentum, done):
    self._v_w = momentum * self._v_w + self._dloss_dw(x, y)
    self._v_b = momentum * self._v_b + self._dloss_db(x, y)
    self.w -= learning_rate * self._v_w
    self.b -= learning_rate * self._v_b
    return self._loss_function(x, y)

  def _loss_function(self, x, y):
    distance = 0.5 * self.w.dot(self.w)
    penalty = np.sum(np.maximum(0, 1 - y * self._forward_step(x)))
    return distance + self.C * penalty

  def _dloss_dw(self, x, y):
    ddistance_dw = self.w
    penalty_idx = np.where(y * self._forward_step(x) < 1)[0]
    dpenalty_dw = - y[penalty_idx].dot(x[penalty_idx])
    return ddistance_dw + self.C * dpenalty_dw

  def _dloss_db(self, x, y):
    ddistance_db = 0
    penalty_idx = np.where(y * self._forward_step(x) < 1)[0]
    dpenalty_db = - y[penalty_idx].sum()
    return ddistance_db + self.C * dpenalty_db

  def _forward_step(self, x):
    return x.dot(self.w) + self.b



class SVC(AbstractSVC):
  '''
    SVC implementation based on the primal lagrangian. Support for kernels is
    included by reparametrizing w = x.T.dot(u). dim(u) = N
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
    self._K = None

  def _build(self, x_shape):
    ''' Initialize weights and momentum. '''
    self.u = np.random.randn(x_shape[0])
    self.b = np.array(0.)
    self._v_u = np.zeros(x_shape[0])
    self._v_b = np.array(0.)

  def _backprop(self, x, y, learning_rate, momentum, done):
    if self._K is None:
      self._x = x
      self._y = y
      self._K = self.kernel(x, x)
    self._v_u = momentum * self._v_u + self._dloss_du()
    self._v_b = momentum * self._v_b + self._dloss_db()
    self.u -= learning_rate * self._v_u
    self.b -= learning_rate * self._v_b
    return self._loss_function()

  def _loss_function(self):
    distance = 0.5 * self.u.dot(self._K).dot(self.u)
    penalty = np.sum(np.maximum(0, 1 - self._y * self._forward_step_train()))
    return distance + self.C * penalty

  def _dloss_du(self):
    ddistance_du = self._K.dot(self.u)
    penalty_idx = np.where(self._y * self._forward_step_train() < 1)[0]
    dpenalty_du = - self._y[penalty_idx].dot(self._K[penalty_idx])
    return ddistance_du + self.C * dpenalty_du

  def _dloss_db(self):
    ddistance_db = 0
    penalty_idx = np.where(self._y * self._forward_step_train() < 1)[0]
    dpenalty_db = - self._y[penalty_idx].sum()
    return ddistance_db + self.C * dpenalty_db

  def _forward_step_train(self):
    return self._K.dot(self.u) + self.b

  def _forward_step(self, x):
    return self.kernel(x, self._x).dot(self.u) + self.b
