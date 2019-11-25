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
        self.set_wind(wind)
        # Current state, used when playing a game
        self.state = None

    # Set state. Used when playing a game
    def _set_state(self, state):
        assert state in self.active_states
        self.state = state

    # Set wind and recalculate transition probabilities
    def set_wind(self, wind):
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

    # Play a game and return history of states, actions and associated rewards.
    def play_game(self, initial_state, player):
        assert isinstance(player, AI)  # TODO: define superclass Player, and class Human
        self._set_state(initial_state)
        history = []
        # Repeat until a terminal state is reached
        while True:
            s = self.state
            actions = self.possible_actions[s]
            a = player.move()
            assert a in actions
            # Update the state. In a windy grid, the new state can be random
            if np.random.random() < self._wind:
                # self.state = np.random.choice(list(self.possible_actions[s].values()))
                self.state = list(actions.values())[np.random.choice(len(actions))]
            else:
                self.state = actions[a]
            r = self.reward[self.state]
            # Update game history
            history.append((s,a,r))
            # Stop if new state is terminal state
            if self.state in self.terminal_states:
                self.state = None
                break
        return history

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
        # Random policy initialization {state: action}
        self.policy = dict()
        for s,actions in env.possible_actions.items():
            self.policy[s] = np.random.choice(list(actions.keys()))
        # State values initialization {state: value}
        self.value = dict()
        for s in env.states:
            self.value[s] = 0
        # Q function initialization {(state, action): (value, n)}
        # n is the number of observations, used to update the mean
        self.Q = dict()
        for s,actions in env.possible_actions.items():
            self.Q[s] = dict()
            for a in actions.keys():  # actions is a dict()
                self.Q[s][a] = (0,0)

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
    def learn_godlike(self):
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

    # Find optimal policy by playing the game multiple times.
    # A Monte Carlo approach is used to updated the estimates of value function
    # Q(s,a), and the optimal policy is found by maximising Q.
    def learn(self, n=10000):
        # Repeat until convergence
        delta_history = []
        print('Training progress:')
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
                    self.Q[s][a] = ((q * m + R ) / (m + 1), m + 1)
            # Update policy in-line, ie. after each update of Q. This saves one
            # optimization loop. In principle, we should first find Q (inner
            # optimization loop), and only then update the policy. And we would
            # have an outer optimization loop to find the optimal policy. Such
            # an optimization algorithm would have very slow convergence.
            # We also update the state value function along with the policy.
            value_change = 0
            policy_change = False
            for s in self.policy.keys():
                best_a = max(self.Q[s], key=self.Q[s].get)  # find a|s s.t. v=Q[s][a] is max
                best_v = self.Q[s][best_a][0]
                value_change = max(value_change, np.abs(best_v - self.value[s]))
                if best_a != self.policy[s]: policy_change = True
                self.policy[s] = best_a
                self.value[s] = best_v
            # Check if we can stop learning (Nope: it stops too early with this algorithm)
            delta_history.append(value_change)
            # if value_change < self.delta and policy_change == False:
            #     print('Training completed after ' + str(i + 1) \
            #         + ' games (value change = ' + str(value_change) + ').')
            #     break
        return delta_history

    # Epsilon-greedy move, used when playing a game
    def move(self):
        if np.random.random() < self.eps:
            return np.random.choice(list(self.env.possible_actions[self.env.state].keys()))
        else:
            return self.policy[self.env.state]

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
        print('Training AI: ' + name)
        # plt.plot(ai.learn_godlike())
        plt.plot(ai.learn())
        plt.title('Convergence')
        plt.show()
        print('Value function:')
        ai.print_value()
        print('Optimal policy:')
        ai.print_policy()
        print()
