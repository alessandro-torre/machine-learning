# Reference: Practical Deep Reinforcement Learning Approach for Stock Trading.
# https://arxiv.org/abs/1811.07522

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
    self._W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
    self._b = np.zeros(output_dim)
    # Momentum
    self._vW = 0
    self._vb = 0
    # Loss history
    self.losses = []

  def predict(self, X):
    assert len(X.shape) == 2
    return X.dot(self._W) + self._b

  def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
    ''' Do one step of gradient descent. '''
    assert len(X.shape) == 2
    n = np.prod(Y.shape)
    # Calculate gradient of squared error
    Yhat = self.predict(X)
    gW = 2 / n * X.T.dot(Yhat - Y)
    gb = 2 / n * (Yhat - Y).sum(axis=0)
    # Update momentum and parameters
    self._vW = momentum * self._vW - learning_rate * gW
    self._vb = momentum * self._vb - learning_rate * gb
    self._W += self._vW
    self._b += self._vb
    # Append and return loss
    loss = np.mean((Y - Yhat)**2)
    self.losses.append(loss)
    return loss



class Scaler:
  ''' Simple scaler to transform states before feeding them to LinearModel. '''
  def __init__(self):
    self._mean = 0
    self._std = 1

  def fit(self, data, initial_investment):
    mean_price = np.mean(data, axis=0)
    std_price = np.sqrt(np.var(data, axis=0))
    mean_cash = initial_investment / 2.
    std_cash = initial_investment / 2.
    mean_stock = initial_investment / mean_price
    std_stock = initial_investment * std_price / mean_price ** 2
    self._mean = np.append(np.append(mean_cash, mean_stock), mean_price)
    self._std = np.append(np.append(std_cash, std_stock), std_price)

  def transform(self, state):
    return (state - self._mean) / self._std



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

  def __init__(self, data, initial_investment):
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
    return self._get_state()

  def _get_value(self):
    return self.cash + self.stock.dot(self.price)

  # Perform the action an the beginning of current time period, then move up in time.
  def step(self, action):
    prev_value = self._get_value()
    # Trade to execute the action.
    self._trade(action)
    # assert self._get_value() == prev_value  # Check portfolio value conservation
    # Move to next time period
    self.t += 1
    self.price = self._history_prices[self.t]
    self.done = self.t == self._t_max
    # Get reward and new state
    reward = self._get_value() - prev_value
    state = self._get_state()
    return state, reward

  def _trade(self, action):
    # Transform action index in the action space to action vector in the action list.
    assert action in self.action_space
    action = self._action_list[action]
    # Determine which stocks to buy and which to sell
    sell_idx = list()
    buy_idx = list()
    for i,a in enumerate(action):
      if a == self.actions['sell']:
        sell_idx.append(i)
      elif a == self.actions['buy']:
        buy_idx.append(i)
    # First generate cash from sales, then buy as much as you can
    for i in sell_idx:
      self.cash += self.price[i] * self.stock[i]
      self.stock[i] = 0
    # Buy each stock at a time until there is enough cash available
    while buy_idx:
      for i in buy_idx:
        if self.cash >= self.price[i]:
          self.cash -= self.price[i]
          self.stock[i] += 1
        else:
          buy_idx.remove(i)

  # Feed data to agent until done and return the history of reward per time step.
  def sim(self, agent, training_mode=False):
    assert isinstance(agent, QNAgent)
    assert self.action_space == agent.action_space
    rewards = np.zeros(self._t_max)
    state = self._reset()
    for t in range(self._t_max):
      action = agent.act(state, training_mode)
      next_state, reward = self.step(action)
      if training_mode:
        agent.train(self.done, state, action, reward, next_state)
      state = next_state
      rewards[t] = reward
    return rewards

  # Estimate the average reward from taking random actions. (SLOW)
  # def average_return(self, n=10000):
  #   history = np.zeros(self._t_max)
  #   for i in range(n):
  #     self._reset()
  #     for t in range(self._t_max):
  #       _, reward = self.step(action=np.random.choice(self.action_space))
  #       history[t] += reward
  #   return history / n

  # Estimate the average return of investing equally in cash and each stock.
  def get_average_return(self):
    return_per_stock = self._history_prices[1:] / self._history_prices[:-1] - 1
    average_return = np.sum(return_per_stock, axis=1) / (self._n_stocks + 1)
    return self.initial_investment * average_return



class QNAgent:
  ''' (Non-deep) Q Network Agent, using Q Reinforcement Learning and no hidden layers.'''
  def __init__(self, state_size, action_space, gamma=0.95, eps=1.0, eps_min=0.01, eps_decay=0.995):
    self.state_size = state_size  # number of variables to describe the state
    self.action_space = action_space  # possible actions
    self.model = LinearModel(state_size, len(action_space))  # regression model used for Q-learning
    self.gamma = gamma  # discounting factor
    self.eps = eps  # epsilon-greedy exploration probability
    self.eps_min = eps_min
    self.eps_decay = eps_decay
    self.scaler = Scaler()  # state scaler used to normalize data for training

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
    # Decaying epsilon
    if self.eps > self.eps_min:
      self.eps *= self.eps_decay
    # Normalize and center input
    state = self.scaler.transform(state).reshape(1,-1)
    next_state = self.scaler.transform(next_state).reshape(1,-1)
    # Default: Same target as current model output (no learning)
    target = self.model.predict(state)  # shape: (1, n_actions)
    # For the taken action, target is total future reward
    if done:
      target[0, action] = reward
    else:
      target[0, action] = reward + self.gamma * self.model.predict(next_state).max()
    # Do one step of sgd. The input variable is simply the state.
    # We do not need to specify the action taken, as this is contained in the
    # target. Note that the model is not learning to take the correct action,
    # but to approximate the value function, ie. the value of taking the action
    # in the given state. The best action is then chosen by maximising the value
    # function, conditionally on the state.
    return self.model.sgd(state, target)

  # Use the trained model to choose the best action, given state.
  # Do espilon-greedy exploration only if in training mode.
  def act(self, state, training_mode=False):
    if training_mode and np.random.random() < self.eps:
      return np.random.choice(self.action_space)
    else:
      # Note that we are passing one state of shape (1, state_size), so
      # model.predict return array of shape (1, n_actions). Therefore, we need
      # to flatten the output and take the first element.
      state = self.scaler.transform(state).reshape(1,-1)
      return np.argmax(self.model.predict(state)[0])

  def save(self, filepath):
    np.savez(filepath, W=self.model._W, b=self.model._b, \
      mean=self.scaler._mean, std=self.scaler._std, eps=self.eps)

  def load(self, filepath):
    npz = np.load(filepath)
    self.model._W = npz['W']
    self.model._b = npz['b']
    self.scaler._mean = npz['mean']
    self.scaler._std = npz['std']
    self.eps = npz['eps']



if __name__ == '__main__':

  # Get and plot time series of stock values
  df = pd.read_csv('data/aapl_msi_sbux.csv')
  # df.plot()
  # plt.show()

  # Split into train and test data
  TRAIN_TEST_SPLIT = 0.5
  data = df.values
  test_idx = int(TRAIN_TEST_SPLIT * len(data))
  train_data = data[:test_idx]
  test_data = data[test_idx:]

  # Initialise environment and agents. Benchmark agent will not be trained, so
  # it will take random actions.
  initial_investment = 20000
  env = MultiStockEnv(train_data, initial_investment)
  agent = QNAgent(env.state_size, env.action_space)

  # Try importing pre-trained weights, otherwise train.
  EPOCHS = 2000
  try:
    filepath = 'models/trader_' + str(EPOCHS) + '.npz'
    filepath2 = 'models/trader_returns/returns_train_' + str(EPOCHS) + '.npy'
    agent.load(filepath)
    print('AI weights restored from ' + filepath)
    train_history = np.load(filepath2)
    print('Train returns restored from ' + filepath2)
  except IOError:
    print(filepath + ' not found, AI will be trained.')
    # Fit the scaler to the data
    agent.scaler.fit(train_data, initial_investment)
    # Train agent on train_data
    train_history = list()  # to store total returns after each sim
    t0 = datetime.now()
    for i in range(EPOCHS):
      portfolio_return = env.sim(agent, training_mode=True).sum()
      train_history.append(portfolio_return)
      if i % max(1, int(EPOCHS/100)) == 0:
        print(f'Training step {i+1}: total return {portfolio_return:.2f}')
    dt = datetime.now() - t0
    print(f'Total training time: {dt} seconds.')
    # Save agent parameters (model weights and scaler parameters) to file
    agent.save(filepath)
    # Save train history of portfolio returns to file
    train_history = np.array(train_history)
    np.save(filepath2, train_history)
    # Plot model losses
    # plt.plot(agent.model.losses)
    # plt.title('Training losses')
    # plt.show()

  # Test agent on test_data, and compare performance with benchmark.
  # Note that we are considering returns here, in exceedance of initial_investment.
  env = MultiStockEnv(test_data, initial_investment)
  return_ai = np.cumsum(env.sim(agent, training_mode=False))
  benchmark = np.cumsum(env.get_average_return())

  # Portfolio return on test data with some random action
  # Try import test returns, otherwise calculate
  try:
    filepath3 = 'models/trader_returns/returns_test_' + str(EPOCHS) + '.npy'
    test_history = np.load(filepath3)
    print('Test returns restored from ' + filepath3)
  except IOError:
    print(filepath3 + ' not found, calculate return on test data.')
    test_history = list()
    for i in range(EPOCHS):
      portfolio_return = env.sim(agent, training_mode=True).sum()
      test_history.append(portfolio_return)
      if i % max(1, int(EPOCHS/100)) == 0:
        print(f'Test step {i+1}: total return {portfolio_return:.2f}')
    # Save train history of portfolio returns to file
    test_history = np.array(test_history)
    np.save(filepath3, test_history)

  # Plot
  df = pd.DataFrame({'agent': return_ai, 'benchmark': benchmark})
  df.plot()
  plt.title('Portfolio return over time, test data (without random exploration).')
  plt.show()
  print(f"Average train return: {train_history.mean():.2f}, min: {train_history.min():.2f}, max: {train_history.max():.2f}")
  plt.hist(train_history, bins=20)
  plt.title('Portfolio returns during training')
  plt.show()
  print(f"Average test return: {test_history.mean():.2f}, min: {test_history.min():.2f}, max: {test_history.max():.2f}")
  plt.hist(test_history, bins=20)
  plt.title('Portfolio returns for test data (with random exploration)')
  plt.show()
