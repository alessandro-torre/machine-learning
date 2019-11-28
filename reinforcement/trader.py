# Reference: Practical Deep Reinforcement Learning Approach for Stock Trading.
# https://arxiv.org/abs/1811.07522

#TODO: scaler (try DIY)
#TODO: decaying eps (with time?)
#TODO: save and load weights

import numpy as np
import pandas as pd
import itertools
from datetime import datetime
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt


class LinearModel:
  ''' Linear regression model using stochastic gradient descent. '''
  def __init__(self, input_dim, output_dim):
    # Parameters
    self._W = np.random.randn(input_dim, output_dim) / input_dim
    self._b = np.zeros(output_dim)
    # Momentum
    self._vW = 0
    self._vb = 0
    # Error history
    self.error_history = []

  def predict(self, X):
    assert len(X.shape) == 2
    return X.dot(self._W) + self._b

  def sgd(self, X, Y, learning_rate=0.1, momentum=0.99):
    ''' Do one step of gradient descent. '''
    assert len(X.shape) == 2
    n = np.prod(Y.shape)
    # Calculate gradient of squared error
    Yhat = self.predict(X)
    self.error_history.append(np.mean((Y - Yhat)**2))
    gW = 2 / n * X.T.dot(Y - Yhat)
    gb = 2 / n * (Y - Yhat).sum(axis=0)
    # Update momentum and parameters
    self._vW = momentum * self._vW - learning_rate * gW
    self._vb = momentum * self._vb - learning_rate * gb
    self._W += self._vW
    self._b += self._vb



class MultiStockEnv:
  '''
  Define the trading environment loosely following the Open AI Gym API.
  In particular:
    state = env.reset()
    while not done:
      action = env.action_space.sample()  # this takes random actions
      action = agent.act(state)  # this action from the agent
      reward = env.step(action)  # this one simplified wrt. OpenAI Gym
  '''
  # 3 actions per stock: sell (all), hold, buy.
  actions = {'sell': 0, 'hold':1, 'buy':2}
  _n_actions = len(actions)

  def __init__(self, data, initial_investment=10000):
    self._t_max, self._n_stocks = data.shape
    self._t_max -= 1
    self._history_prices = data
    self.initial_investment = initial_investment
    self._reset()
    # State vars: cash + price & quantity of each stock
    self.state_size = 1 + 2 * self._n_stocks
    # Generate all possible combinations of actions on the stocks.
    # The action list is the internal representation of actions.
    # The action space is the API for the action list (a list of indices).
    self._action_list = list(map(list,
      itertools.product(range(self._n_actions), repeat=self._n_stocks)))
    self.action_space = range(self._n_actions ** self._n_stocks)
    assert len(self.action_space) == len(self._action_list)

  # Return the state representation as the array:
  # [cash, n_stock_1, ..., n_stock_m, value_stock_1, ..., value_stock_m]
  def _get_state(self):
    return np.append(np.append(self.cash, self.stock), self.price)

  def _reset(self):
    self.done = False
    self.t = 0
    self.cash = self.initial_investment
    self.stock = np.zeros(self._n_stocks)
    self.price = self._history_prices[self.t]
    self.value = self._get_value()
    return self._get_state()

  def _get_value(self):
    return self.cash + self.stock.dot(self.price)

  # Perform the action an the beginning of current time period, then move up in time.
  def step(self, action):
    # Transform action index in the action space to action vector in the action list.
    assert action in self.action_space
    action = self._action_list[action]
    # First generate cash from sales, then buy as much as you can
    sell_idx = list()
    buy_idx = list()
    for i,a in enumerate(action):
      if a == self.actions['sell']:
        sell_idx.append(i)
      elif a == self.actions['buy']:
        buy_idx.append(i)
    for i in sell_idx:
      self.cash += self.price[i] * self.stock[i]
      self.stock[i] = 0
    # Buy uniformly across stocks, ie. buy one share of each at a time until possible
    enough_cash = True
    while enough_cash:
      if not buy_idx: break
      for i in buy_idx:
        if self.cash > self.price[i]:
          self.cash -= self.price[i]
          self.stock[i] += 1
        else:
          enough_cash = False
    # Move to next time period
    self.t += 1
    self.price = self._history_prices[self.t]
    self.done = self.t == self._t_max
    # Calculate the reward
    previous_value = self.value
    self.value = self._get_value()
    reward = self.value - previous_value
    state = self._get_state()
    return state, reward

  # Process data until done and return return the history of reward per time step.
  def sim(self, agent, training_mode=False):
    assert self.action_space == agent.action_space
    history = []
    state = self._reset()
    for t in range(self._t_max):
      action = agent.act(state, training_mode)
      next_state, reward = self.step(action)
      if training_mode:
        agent.train(self.done, state, action, reward, next_state)
      state = next_state
      history.append(reward)
    return np.array(history)



class QNAgent:
  ''' (Non-deep) Q Network Agent, using Q Reinforcement Learning and no hidden layers.'''
  def __init__(self, state_size, action_space, gamma=0.95, eps=0.10):
    self.state_size = state_size  # number of variables to describe the state
    self.action_space = action_space  # possible actions
    self.model = LinearModel(state_size, len(action_space))  # regression model used for Q-learning
    self.gamma = gamma  # discounting factor
    self.eps = eps  # epsilon-greedy exploration probability\

  # Q-learning: the regression model is trained to target total future reward,
  # i.e. immediate reward + (discounted) next state value, where the latter is
  # approximated by using of the model itself (which is being trained, hence it
  # is an approximation). Since a state value is its maximum return under all
  # possible actions, and the model learns to calculate state returns under all
  # possible actions, we need the maximum model output for the next state.
  # Since we want to use a generic regression model, we need to pass targets
  # for the complete action space. The trick is to pass current model output for
  # the actions that are not taken, so that error and gradient are zero along
  # those dimensions of the action space.
  def train(self, done, state, action, reward, next_state):
    # Default: Same target as current model output (no learning)
    state = state.reshape(1,-1)
    target = self.model.predict(state)  # shape: (1, n_actions)
    # Target is total future reward for the taken action
    if done:
      target[0, action] = reward
    else:
      target[0, action] = reward \
        + self.gamma * self.model.predict(next_state.reshape(1,-1)).max()
    # Do one step of sgd. The input variable is simply the state.
    # We do not need to specify the action taken, as this is contained in the
    # target. Note that the model is not learning to take the correct action,
    # but to approximate the value function, ie. the value of taking the action
    # in the given state. The best action is then chosen by maximising the value
    # function, conditionally on the state.
    self.model.sgd(state, target)

  # Use the trained model to choose the best action, given state.
  # Do espilon-greedy exploration only if in training mode.
  def act(self, state, training_mode=False):
    if training_mode and np.random.random() < self.eps:
      return np.random.choice(self.action_space)
    else:
      # Note that we are passing one state of shape (1, state_size), so
      # model.predict return array of shape (1, n_actions). Therefore, we need
      # to flatten the output and take the first element.
      return np.argmax(self.model.predict(state.reshape(1,-1))[0])


if __name__ == '__main__':

  # Get and plot time series of stock values
  df = pd.read_csv('aapl_msi_sbux.csv')
  # df.plot()
  # plt.show()

  # Split into train and test data
  TRAIN_TEST_SPLIT = 0.8
  data = df.values
  test_idx = int(TRAIN_TEST_SPLIT * len(data))
  train_data = data[:test_idx]
  test_data = data[test_idx:]

  # Initialise environment and agents. Benchmark agent will not be trained, so
  # it will take random actions.
  env = MultiStockEnv(train_data)
  agent = QNAgent(env.state_size, env.action_space)
  benchmark = QNAgent(env.state_size, env.action_space)

  # Train agent on train_data
  EPOCHS = 1000
  train_history = list()  # to store total returns after each sim
  t0 = datetime.now()
  for i in range(EPOCHS):
    sim_history = env.sim(agent, training_mode=True)
    sim_return = sim_history.sum()
    train_history.append(sim_return)
    if i % int(EPOCHS/100) == 0:
      print(f'Training step {i+1}: total reward {sim_return:.2f}')
  dt = datetime.now() - t0
  print(f'Total training time: {dt} seconds.')
  plt.plot(train_history)
  plt.show()

  # Test agent on test_data, and compare performance with benchmark.
  env = MultiStockEnv(test_data)
  history_ai = env.sim(agent)
  history_bm = env.sim(benchmark)
  history = pd.DataFrame({'agent': history_ai, 'benchmark': history_bm})
  history.plot()
  plt.show()
