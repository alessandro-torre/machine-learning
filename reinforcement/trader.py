# Practical Deep Reinforcement Learning Approach for Stock Trading
# https://arxiv.org/abs/1811.07522

import numpy as np
import pandas as ps
import itertools as it

class LinearModel:
  ''' Linear regression model using stochastic gradient descent. '''
  def __init__(self, input_dim, n_actions):
    # Parameters
    self._W = np.random.randn(input_dim, n_actions) / input_dim
    self._b = np.zeros(n_actions)
    # Momentum
    self._vW = 0
    self._vb = 0
    # Error history
    self.error_history = []

  def predict(self, X):
    assert dim(X) == 2
    return X.dot(self._W) + self._b

  def sgd(self, X, Y, learning_rate=0.1, momentum=0.99):
    ''' Do one step of gradient descent. '''
    assert dim(X) == 2
    n = np.prod(Y.shape)

    # Calculate gradient of squared error
    Yhat = self.predict(X)
    self.error_history.append(np.mean((Y - Yhat)**2))
    gW = 2 / n * X.T.dot(Y - Yhat)
    gb = 2 / n * (Y - Yhat).sum(axis=0)

    # Update momentum and parameters
    self._vW = self.momentum * self._vW - self.learning_rate * gW
    seld._vb = self.momentum * self._vb - self.learning_rate * gb
    self._W += self._vW
    self._b += self._vb



class MultiStockEnv:
  def __init__(self, data, initial_investment=10000):
    self._t_max, self._n_stocks = data.shape
    self.history_prices = data
    self.initial_investment = initial_investment
    self._reset()

    # State vars: cash + price & quantity of each stock
    self._state_dim = 1 + 2 * self._n_stocks

    # 3 actions (buy, hold, sell) per stock
    self._action_space = range(3 * self._n_stocks)
    self._action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

  def _reset(self):
    self._t = 0
    self.cash = initial_investment
    self.stocks = np.zeros(self._n_stocks)
    self.prices = self.history_prices[self._t]

  def one_step(self, action):


  def sim(self, agent):
    self._reset()
    for t in range(self._t_max):
      a = agent.trade()
      pass


class QNAgent:
  ''' (Non-deep) Q Network Agent, using Q Reinforcement Learning and no hidden layers.'''
  def __init__(self, model):
    self.model = LinearModel()

  def train(self):
    pass

  def trade(self):
    pass



if __name__ == '__main__':

  df = pd.read_csv('aapl_msi_sbux.csv')
  data = df.values

  EPOCHS = 2000
  # TRAIN_MODE = False

  env = MultiStockEnv(train_data)
  agent = QNAgent()

  for i in range(EPOCHS):
    pass
