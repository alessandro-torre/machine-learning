# Reference: Practical Deep Reinforcement Learning Approach for Stock Trading.
# https://arxiv.org/abs/1811.07522

import os
import numpy as np
import pandas as pd
import tensorflow as tf
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

  def train_on_batch(self, X, Y, learning_rate=0.01, momentum=0.9):
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



class ReplayBuffer:
  def __init__(self, state_size, memory_size):
    self.pointer = 0
    self.memory_used = 0
    self.memory_size = memory_size
    self.state = np.zeros([memory_size, state_size], dtype=np.float32)
    self.next_state = np.zeros([memory_size, state_size], dtype=np.float32)
    self.action = np.zeros(memory_size, dtype=np.uint8)
    self.reward = np.zeros(memory_size, dtype=np.float32)
    self.done = np.zeros(memory_size, dtype=bool)

  def store(self, state, action, next_state, reward, done):
    self.state[self.pointer] = state
    self.action[self.pointer] = action
    self.next_state[self.pointer] = next_state
    self.reward[self.pointer] = reward
    self.done[self.pointer] = done
    self.pointer = (self.pointer + 1) % self.memory_size
    self.memory_used = min(self.memory_used + 1, self.memory_size)

  def read(self, batch_size):
    batch_size = min(self.memory_used, batch_size)
    idx = np.random.randint(low=0, high=self.memory_used, size=batch_size)
    return {'state': self.state[idx],
            'action': self.action[idx],
            'next_state': self.next_state[idx],
            'reward': self.reward[idx],
            'done': self.done[idx]}



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

  def set_agent(self, agent):
    assert isinstance(agent, DQNAgent)
    assert self.action_space == agent.action_space
    self.agent = agent

  # Feed data to agent until done and return the history of reward per time step.
  def sim(self, training_mode=False):
    rewards = np.zeros(self._t_max)
    state = self._reset()
    for t in range(self._t_max):
      action = self.agent.act(state, training_mode)
      next_state, reward = self.step(action)
      if training_mode:
        agent.train_on_batch(state, action, next_state, reward, self.done)
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



class DQNAgent:
  ''' Deep Q Network Agent, using Deep Q Reinforcement Learning.'''
  def __init__(self, state_size, action_space, gamma=0.95, eps=1.0,
              eps_min=0.01, eps_decay=0.995, memory_size=500, batch_size=32):
    self.state_size = state_size  # number of variables to describe the state
    self.action_space = action_space  # possible actions
    self.gamma = gamma  # discounting factor
    self.eps = eps  # epsilon-greedy exploration probability
    self.eps_min = eps_min
    self.eps_decay = eps_decay
    self.batch_size = batch_size
    self.scaler = Scaler()  # state scaler used to normalize data for training
    self.buffer = ReplayBuffer(state_size, memory_size)
    # self.model = LinearModel(state_size, len(action_space))
    self.model = self._init_model(state_size, len(action_space))

  def _init_model(self, input_dim, output_dim):
    i = tf.keras.layers.Input(shape=(input_dim,))
    x = i
    x = tf.keras.layers.Dense(20, activation='relu')(x)
    x = tf.keras.layers.Dense(output_dim)(x)
    model = tf.keras.Model(i, x)
    model.compile(optimizer='adam', loss='mse')
    return model

  def call(self, inputs):
    x = self.hidden_layer(inputs)
    x = self.output_layer(x)
    return x

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
  def train_on_batch(self, state, action, next_state, reward, done):
    # Decaying epsilon
    if self.eps > self.eps_min:
      self.eps *= self.eps_decay
    # Normalize and center input
    state = self.scaler.transform(state).reshape(1,-1)
    next_state = self.scaler.transform(next_state).reshape(1,-1)
    # Update the replay buffer, then read from it
    self.buffer.store(state, action, next_state, reward, done)
    batch = self.buffer.read(self.batch_size)
    state = batch['state']
    action = batch['action']
    next_state = batch['next_state']
    reward = batch['reward']
    done = batch['done']
    # Default: Same target as current model output (ie. no learning).
    target = self.model.predict(state)  # shape: (batch_size, n_actions)
    # For each taken action, set target to be the immediate reward.
    target[range(len(action)), action] = reward
    # For each taken action where we are not done, add discounted future return,
    # estimated by current value function (ie. current model weights).
    if not done.all():
      not_done = np.where(done == False)
      target[not_done, action[not_done]] += \
        self.gamma * self.model.predict(next_state[not_done]).max(axis=1)
    # Do one step of sgd. The input variable is simply the state.
    # We do not need to specify the action taken, as this is contained in the
    # target. Note that the model is not learning to take the correct action,
    # but to approximate the value function, ie. the value of taking the action
    # in the given state. The best action is then chosen by maximising the value
    # function, conditionally on the state.
    return self.model.train_on_batch(state, target)

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
  agent = DQNAgent(env.state_size, env.action_space)
  env.set_agent(agent)

  # Try importing pre-trained weights, otherwise train.
  EPOCHS = 1000
  n_msg = 1000
  models_folder = 'models/'
  returns_folder = models_folder + 'trader_returns/'
  if not os.path.exists(models_folder): os.makedirs(models_folder)
  if not os.path.exists(returns_folder): os.makedirs(returns_folder)
  try:
    filepath = models_folder + 'trader_' + str(EPOCHS) + '.npz'
    file_returns_train = returns_folder + 'returns_train_' + str(EPOCHS) + '.npy'
    agent.load(filepath)
    print('AI weights restored from ' + filepath)
    train_history = np.load(file_returns_train)
    print('Train returns restored from ' + file_returns_train)
  except IOError:
    print(filepath + ' not found, AI will be trained.')
    # Fit the scaler to the data
    agent.scaler.fit(train_data, initial_investment)
    # Train agent on train_data
    train_history = list()  # to store total returns after each sim
    t0 = datetime.now()
    for i in range(EPOCHS):
      t1 = datetime.now()
      portfolio_return = env.sim(training_mode=True).sum()
      train_history.append(portfolio_return)
      if i % max(1, EPOCHS//n_msg) == 0:
        print(f'Training step {i+1} [{datetime.now() - t1}]: '
              f'total return {portfolio_return:.2f}')
    print(f'Total training time: {datetime.now() - t0}.')
    # Save agent parameters (model weights and scaler parameters) to file
    agent.save(filepath)
    # Save train history of portfolio returns to file
    train_history = np.array(train_history)
    np.save(file_returns_train, train_history)
    # Plot model losses
    # plt.plot(agent.model.losses)
    # plt.title('Training losses')
    # plt.show()

  # Test agent on test_data, and compare performance with benchmark.
  # Note that we are considering returns here, in exceedance of initial_investment.
  env = MultiStockEnv(test_data, initial_investment)
  env.set_agent(agent)
  return_ai = np.cumsum(env.sim(training_mode=False))
  benchmark = np.cumsum(env.get_average_return())

  # Portfolio return on test data with some random action
  # Try import test returns, otherwise calculate
  try:
    file_returns_test = returns_folder + 'returns_test_' + str(EPOCHS) + '.npy'
    test_history = np.load(file_returns_test)
    print('Test returns restored from ' + file_returns_test)
  except IOError:
    print(file_returns_test + ' not found, calculate return on test data.')
    test_history = list()
    for i in range(EPOCHS):
      portfolio_return = env.sim(training_mode=True).sum()
      test_history.append(portfolio_return)
      if i % max(1, EPOCHS//n_msg) == 0:
        print(f'Test step {i+1}: total return {portfolio_return:.2f}')
    # Save train history of portfolio returns to file
    test_history = np.array(test_history)
    np.save(file_returns_test, test_history)

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
