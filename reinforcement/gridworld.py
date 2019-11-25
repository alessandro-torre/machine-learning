import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt

class Environment:
    actions = {'U':(-1,0), 'D':(+1,0), 'L':(0,-1), 'R':(0,+1)}

    # Set the state grid and associated rewards.
    # Convention: reward None denotes unaccessible states.
    def __init__(self, reward_matrix, terminal_states, wind=0.):
        assert isinstance(terminal_states, set)
        self.terminal_states = terminal_states
        self.shape = reward_matrix.shape
        # Construct dict of reachable states and associated rewards
        self.states = set()  # set of all grid states
        self.reward = dict()  # dict of states and associated rewards
        for i in range(reward_matrix.shape[0]):
            for j in range(reward_matrix.shape[1]):
                # Skip states marked as np.nan (i.e. reward=None)
                if not np.isnan(reward_matrix[i,j]):
                    self.states.add((i,j))
                    self.reward[(i,j)] = reward_matrix[i,j]
        # Construct dict of states and allowed actions.
        # The resulting endstate is deterministic (i.e. without random 'wind').
        self.active_states = set() # set of non terminal states
        self.possible_actions = dict()  # dict of allowed actions per state
        for s in self.states:
            if s not in self.terminal_states:
                self.active_states.add(s)
                self.possible_actions[s] = dict()
                for a,x in self.actions.items():
                    s2 = (s[0] + x[0], s[1] + x[1])  # new state s'
                    if s2 in self.states:
                        self.possible_actions[s][a] = s2
        # Calculate transition probabilities
        self._set_wind(wind)
        # Current state, used when playing a game
        self.state = None

    # Set state
    def _set_state(self, state):
        assert state in self.active_states
        self.state = state

    # Set wind and recalculate transition probabilities
    def _set_wind(self, wind):
        assert wind >= 0 and wind <= 1
        self._wind = wind
        self._trans_prob = dict()
        for s,actions in self.possible_actions.items():
            wind_prob = wind / len(actions)
            for a_target in actions.keys():
                self._trans_prob[s,a_target] = dict()
                for a_actual,s2 in actions.items():
                    self._trans_prob[s,a_target][s2] = wind_prob + \
                        ((1 - wind) if a_actual == a_target else 0)

    # Set initial state and, optionally, change wind.
    def start_game(self, initial_state, wind=None):
        if wind is not None: self._set_wind(wind)
        self._set_state(initial_state)

    # Play a game and return history of states, actions and associated rewards.
    def play_game(self, initial_state, player):
        assert isinstance(player, AI)  # TODO: define superclass Player, and class Human
        self.start_game(initial_state)
        history = []
        # Repeat until a terminal state is reached
        while not self.is_gameover():
            s = self.state
            a = player.move()
            _, r = self.play_action(a)  # apply the move and update the environment
            # Update game history
            history.append((s,a,r))
        return history

    # Play action, update environment and return reward.
    # Used by play_game(), and directly by fully online training methods.
    def play_action(self, action):
        actions = self.possible_actions[self.state]
        assert action in actions
        # Update the state. In a windy grid, the new state can be random
        if np.random.random() < self._wind:
            self.state = list(actions.values())[np.random.choice(len(actions))]
        else:
            self.state = actions[action]
        return self.state, self.reward[self.state]

    def is_gameover(self):
        return self.state in self.terminal_states

    def print_trans_prob(self):
        print(self._print_d(self._trans_prob))

    def print_possible_actions(self):
        print(self._print_d(self.possible_actions))

    # Recursive function to print nested dictionary in a readable way
    def _print_d(self, dd, lvl=0):
        if type(dd) != dict:
            return str(dd)
        else:
            res = []
            for x,y in dd.items():
                res.append(''.join((' '*4*lvl, str(x),': ',self._print_d(y, lvl+1))))
            return '\n'*(lvl>0) + '\n'.join(res)

    def print_reward(self):
        for i in range(self.shape[0]):
            print('-' * 6 * self.shape[1])
            for j in range(self.shape[1]):
                r = self.reward.get((i,j), np.nan)
                if np.isnan(r):
                    print('     |', end='')
                elif r >= 0:
                    print(' %.2f|' % r, end='')
                else:
                    print('%.2f|' % r, end='') # -ve sign takes up an extra space
            print()
        print('-' * 6 * self.shape[1])


class AI:
    def __init__(self, env, gamma=1, eps=0.1, delta=1e-5):
        assert isinstance(env, Environment)
        assert gamma >= 0 and gamma <= 1
        assert eps >= 0 and eps <= 1
        self.env = env  # game environment
        self.gamma = gamma  # one-period discounting factor for future rewards
        self.eps = eps  # prob to explore (epsilon-greedy strategy)
        self.delta = delta  # convergence threshold
        self.reset()
        # List of available training methods
        self._train = {
            'god'  : self._learn_god,
            'mc'   : self._learn_mc,
            'sarsa': self._learn_sarsa,
            'ql'   : self._learn_ql
        }
        self.training_methods = list(self._train.keys())

    # Reset policy, value function, and Q.
    def reset(self):
        # Random policy initialization {state: action}
        self.policy = dict()
        for s,actions in self.env.possible_actions.items():
            self.policy[s] = np.random.choice(list(actions.keys()))
        # State values initialization {state: value}
        self.value = dict()
        for s in self.env.states:
            self.value[s] = 0
        # Q function initialization {(state, action): (value, n)}
        # n is the number of observations, used to update the mean
        self.Q = dict()
        for s,actions in self.env.possible_actions.items():
            self.Q[s] = dict()
            for a in actions.keys():  # actions is a dict()
                self.Q[s][a] = (0,0)

    # Wrapper method to train the AI
    def learn(self, method='mc', n=None):
        assert method in self.training_methods
        print('Starting training with method: ' + method)
        if n is None:
            return self._train[method]()
        else:
            assert type(n) is int and n > 0
            return self._train[method](n)

    # Determine states value and find optimal policy assuming we are in god mode
    # ie. we know the grid's transition probabilities. This allows the agent
    # to learn without really playing the game. It is like cheating, since a
    # real agent would not know the transition probabilities but it would need
    # instead to play the game multiple times, in order to learn the average
    # outcome of each action in each state.
    #
    # This function is useful to easily solve the prediction problem (policy
    # evaluation) and the control problem (policy selection), and to benchmark a
    # more realistic reinforcement learning algorithms that does not rely on
    # knowing the transition probabilities, such as a Monte Carlo approach.
    #
    # Optimal policy: for each state, perform the action that maximizes the
    # state value, i.e. the average future reward of the state.
    # State value = the best average future reward of the state
    #             = value of the state unde the optimal policy.
    def _learn_god(self, _=None):
        # Repeat until convergence.
        delta_history = []
        i = 0
        while True:
            i += 1
            value_change = 0
            policy_change = False
            # Loop through all active states
            for s,actions in self.env.possible_actions.items():
                best_v = float('-inf')
                best_a = None
                # Loop through all possible actions to determine the value.
                for a in actions.keys():  # actions is a dict()
                    v = 0  # we will accumulate
                    # Add the average future reward for the selected action.
                    for s2,p in self.env._trans_prob[(s,a)].items():
                        v += p * (self.env.reward[s2] + self.gamma * self.value[s2])
                    if v > best_v:
                        best_v, best_a = (v, a)
                value_change = max(value_change, np.abs(best_v - self.value[s]))
                # Update policy and state value
                if best_a != self.policy[s]: policy_change = True
                self.policy[s] = best_a
                self.value[s] = best_v
            # Check if we can stop learning
            delta_history.append(value_change)
            if value_change < self.delta and policy_change == False:
                print('Training completed after ' + str(i) \
                    + ' games (value change = ' + str(value_change) + ').')
                break
        return delta_history

    # Find optimal policy by playing the game multiple times and using a Monte
    # Carlo (MC) approach to update the estimates of value function Q(s,a).
    # The optimal policy is found by maximising Q(s,a) wrt. a
    # This is a semi-online algorithm, since updates happen only at the end of
    # each game, and not after each move.
    def _learn_mc(self, n=1000):
        delta_history = []
        print('Training progress:')
        # Repeat for n times
        for i in range(n):
            # Print simulation progress
            if (i + 1) % (n // 10) == 0: print(str(100 * (i + 1) // n) + '%')
            initial_state = list(self.env.active_states)[np.random.choice(len(self.env.active_states))]
            history_r = self.env.play_game(initial_state, player=self)
            # Calculate total future reward per visited state
            R = 0
            history_R = []
            for s,a,r in reversed(history_r):
                R = r + self.gamma * R
                history_R.append((s,a,R))
            # Update Q(s,a)
            seen = set()
            for s,a,R in history_R:
                if (s,a) not in seen:
                    seen.add((s,a))
                    (q,m) = self.Q[s][a]
                    # Update the estimate using a (decaying) learning rate alpha=1/(m+1)
                    self.Q[s][a] = ((q * m + R) / (m + 1), m + 1)
            # Update policy online, ie. after each update of Q. This saves one
            # optimization loop. In principle, we should first find Q (inner
            # optimization loop), and only then update the policy. And we would
            # have an outer optimization loop to find the optimal policy. Such
            # an optimization algorithm would have very slow convergence.
            # We also update the state value function along with the policy.
            delta_history.append(self._update_from_Q())
        return delta_history

    # TD0 is similar to MC, but more online, i.e. we update Q after each move,
    # and not at the end of a game.
    # Difference with MC: we do not calculate use total future returns R from
    # the last game history, but we estimate (or 'bootstrap') it with the
    # (current) next state value (the state value is the total future reward).
    def _learn_td0(self, n=1000):
        delta_history = []
        print('Training progress:')
        # Repeat for n times
        for i in range(n):
            # Print simulation progress
            if (i + 1) % (n // 10) == 0: print(str(100 * (i + 1) // n) + '%')
            s = list(self.env.active_states)[np.random.choice(len(self.env.active_states))]
            self.env.start_game(s)
            a = self.move()
            # Play until gameover
            while not self.env.is_gameover():
                s2, r = self.env.play_action(a)
                # Check if it is already gameover
                if self.env.is_gameover():
                    a2 = None
                    next_q = 0
                else:
                    a2 = self.move()  # required now to update Q, but played later
                    next_q = self.Q[s2][a2][0]
                # Update the estimate using a (decaying) learning rate alpha=1/(m+1)
                q, m = self.Q[s][a]
                self.Q[s][a] = ((q * m + (r + self.gamma * next_q)) / (m + 1), m + 1)
                # Update for next round
                s = s2
                a = a2
            # Update policy and value function.
            delta_history.append(self._update_from_Q())
        return delta_history

    # SARSA is similar to TD0, but even more online, i.e. we update the policy
    # after each move, along with Q.
    # In practice: we do not need to update policy and state values after each
    # move. We can simply choose actions based on argmax(Q), instead of policy.
    def _learn_sarsa(self, n=1000):
        delta_history = []
        print('Training progress:')
        # Repeat for n times
        for i in range(n):
            # Print simulation progress
            if (i + 1) % (n // 10) == 0: print(str(100 * (i + 1) // n) + '%')
            s = list(self.env.active_states)[np.random.choice(len(self.env.active_states))]
            self.env.start_game(s)
            a = self.move()
            # Play until gameover
            while not self.env.is_gameover():
                s2, r = self.env.play_action(a)
                # Check if it is already gameover
                if self.env.is_gameover():
                    a2 = None
                    next_q = 0
                else:
                    # SARSA: We select second action based on maximizing Q, instead of the policy.
                    a2 = self._randomize(self._argmax_Q(s2))
                    next_q = self.Q[s2][a2][0]
                # Update the estimate using a (decaying) learning rate alpha=1/(m+1)
                q, m = self.Q[s][a]
                self.Q[s][a] = ((q * m + (r + self.gamma * next_q)) / (m + 1), m + 1)
                # Update for next round
                s = s2
                a = a2
                self.policy
            # Update policy and value function. Note: we could do it at the end
            # of the training, since we do not use the policy while training.
            # Here we simply want to update the history of changes to value function.
            delta_history.append(self._update_from_Q())
        return delta_history

    # Q-learning is similar to SARSA, but off-policy.
    # We do not necessarily perform the action that we use to update Q.
    def _learn_ql(self, n=1000):
        delta_history = []
        print('Training progress:')
        # Repeat for n times
        for i in range(n):
            # Print simulation progress
            if (i + 1) % (n // 10) == 0: print(str(100 * (i + 1) // n) + '%')
            s = list(self.env.active_states)[np.random.choice(len(self.env.active_states))]
            self.env.start_game(s)
            a = self.move()
            # Play until gameover
            while not self.env.is_gameover():
                s2, r = self.env.play_action(a)
                # Check if it is already gameover
                if self.env.is_gameover():
                    a2 = None
                    next_q = 0
                else:
                    # Q-learning: We select second action based on Q, and no randomization
                    a2 = self._argmax_Q(s2)
                    next_q = self.Q[s2][a2][0]
                    a2 = self._randomize(a2) # Now we randomize
                # Update the estimate using a (decaying) learning rate alpha=1/(m+1)
                q, m = self.Q[s][a]
                self.Q[s][a] = ((q * m + (r + self.gamma * next_q)) / (m + 1), m + 1)
                # Update for next round
                s = s2
                a = a2
                self.policy
            # Update policy and value function. Note: we could do it at the end
            # of the training, since we do not use the policy while training.
            # Here we simply want to update the history of changes to value function.
            delta_history.append(self._update_from_Q())
        return delta_history

    # Update policy and value function based on maximizing Q.
    # Used with MC and SARSA to perform online policy optimization.
    # Return maximum value change and boolean for tracking policy changes.
    def _update_from_Q(self):
        value_change = 0
        # policy_change = False
        for s in self.policy.keys():
            best_a = self._argmax_Q(s)
            best_v = self.Q[s][best_a][0]
            value_change = max(value_change, np.abs(best_v - self.value[s]))
            # if best_a != self.policy[s]: policy_change = True
            self.policy[s] = best_a
            self.value[s] = best_v
        return value_change  # , policy_change

    # Given states , return action a that maximizes Q(s,a)
    def _argmax_Q(self, state):
        assert state in self.env.active_states
        return max(self.Q[state], key=self.Q[state].get)

    # Select action based on policy, with epsilon-greedy random exploration.
    def move(self, explore=True):
        assert self.env.state in self.env.active_states
        a = self.policy[self.env.state]
        if explore: a = self._randomize(a)
        return a

    # Epsilon-greedy exploration strategy
    def _randomize(self, action):
        assert self.env.state in self.env.possible_actions
        assert action in self.env.possible_actions[self.env.state]
        if np.random.random() < self.eps:
            return np.random.choice(list(self.env.possible_actions[self.env.state].keys()))
        else:
            return action

    def print_policy(self):
        for i in range(self.env.shape[0]):
            print('-' * 6 * self.env.shape[1])
            for j in range(self.env.shape[1]):
                r = self.env.reward.get((i,j), np.nan)
                r_s = ' ' if np.isnan(r) else '+' if r >= 0 else '-'
                a = self.policy.get((i,j), r_s)
                print('  %s  |' % a, end='')
            print()
        print('-' * 6 * self.env.shape[1])

    def print_value(self):
        for i in range(self.env.shape[0]):
            print('-' * 6 * self.env.shape[1])
            for j in range(self.env.shape[1]):
                v = self.value.get((i,j), np.nan)
                if np.isnan(v):
                    print('     |', end='')
                elif v >= 0:
                    print(' %.2f|' % v, end='')
                else:
                    print('%.2f|' % v, end='')
            print()
        print('-' * 6 * self.env.shape[1])


if __name__ == '__main__':

    terminal_states={(0,3), (1,3)}
    reward_matrix = np.full((4,6), -0.1)  # penalty for each move
    reward_matrix[0,3] = +1  # target state
    reward_matrix[1,3] = -1  # pitfall state
    # unaccessible states
    reward_matrix[1,1] = reward_matrix[0,4] = reward_matrix[2,4] = None

    envs = {
        'simple': Environment(reward_matrix, terminal_states),
        'windy': Environment(reward_matrix, terminal_states, wind=.4),
        'too windy': Environment(reward_matrix, terminal_states, wind=1)
    }
    print('Rewards:')
    envs['simple'].print_reward()
    # envs['simple'].print_possible_actions()
    # for name,env in envs.items():
    #     print(name + ' transition probabilities:')
    #     env.print_trans_prob()

    # Note that we need to increase eps lo learn effectively in windy environments!
    ais = {
        # 'simple': AI(envs['simple']),
        # 'gamma': AI(envs['simple'], gamma=0.9),  # note the change in value function
        # 'windy': AI(envs['windy'], eps=0.5),  # note the change in value function and policy
        'windy gamma': AI(envs['windy'], gamma=0.9, eps=0.5),  # note the change in value function and policy
        # 'too greedy': AI(envs['simple'], gamma=0), # unable to see only one step further, unable to reach target state
        # 'too windy': AI(envs['too windy'], gamma=0, eps=0.5) # any action has random outcome, unable to learn anything
    }

    for name,ai in ais.items():
        print('AI: ' + name)
        for method in ai.training_methods:
            ai.reset()
            plt.plot(ai.learn(method))
            plt.title('Convergence of training method: ' + method)
            plt.show()
            print('Value function:')
            ai.print_value()
            print('Optimal policy:')
            ai.print_policy()
            print()
