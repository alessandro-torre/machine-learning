import numpy as np

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

    # Set wind and recalculate transition probabilities
    def set_wind(self, wind):
        assert wind >= 0 and wind <= 1
        self._wind = wind
        self.trans_prob = dict()
        for s,actions in self.possible_actions.items():
            wind_prob = wind / len(actions)
            for a_target in actions.keys():
                self.trans_prob[s,a_target] = dict()
                for a_actual,s2 in actions.items():
                    self.trans_prob[s,a_target][s2] = wind_prob + \
                        ((1 - wind) if a_actual == a_target else 0)

    def print_trans_prob(self):
        print(self._print_d(self.trans_prob))

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
    def __init__(self, env, gamma=1, eps=1e-3):
        assert isinstance(env, Environment)
        assert gamma >= 0 and gamma <= 1
        assert eps >= 0 and eps <= 1
        self.env = env
        self.gamma = gamma
        self.eps = eps
        # Random policy initialization
        self.policy = dict()  # dictionary of policy actions per state
        for s,actions in env.possible_actions.items():
            self.policy[s] = list(actions.keys())[np.random.choice(len(actions))]
        # State values initialization
        self.value = dict()  # dictionary of state values
        for s in env.states:
            self.value[s] = 0

    # Update states value and policy.
    # Policy: select the best action per state, defined as the action that
    # maximizes the average future reward.
    # State value: defined equal to the best average future reward.
    def learn(self):
        # Repeat until convergence.
        while True:
            value_change = 0.
            policy_change = False
            # Loop through all active states
            for s,actions in self.env.possible_actions.items():
                best_v = float('-inf')
                best_a = None
                # Loop through all possible actions to determine the value.
                for a in actions.keys():
                    v = 0  # we will accumulate
                    # Add the average future reward for the selected action.
                    for s2,p in self.env.trans_prob[(s,a)].items():
                        v += p * (self.env.reward[s2] + self.gamma * self.value[s2])
                    if v > best_v:
                        best_v, best_a = (v, a)
                # Update policy
                if best_a != self.policy[s]:
                    policy_change = True
                    self.policy[s] = best_a
                # Update state value
                value_change = max(value_change, np.abs(best_v - self.value[s]))
                self.value[s] = best_v
            # Check if we can break the loop
            if value_change < self.eps and policy_change == False: break

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

    reward_matrix = np.full((4,6), -0.1)  # penalty for each move
    reward_matrix[0,3] = +1    # target state
    reward_matrix[1,3] = -1    # pitfall state
    reward_matrix[1,1] = None  # unaccessible state
    terminal_states={(0,3), (1,3)}

    envs = {
        'simple': Environment(reward_matrix, terminal_states),
        'windy': Environment(reward_matrix, terminal_states, wind=.5),
        'too windy': Environment(reward_matrix, terminal_states, wind=1)
    }
    print('Rewards:')
    envs['simple'].print_reward()
    for name,env in envs.items():
        print(name + ' transition probabilities:')
        env.print_trans_prob()

    ais = {
        'simple': AI(envs['simple']),
        'gamma': AI(envs['simple'], gamma=0.9),  # note the change in value function
        'windy': AI(envs['windy']),  # note the change in value function and policy
        'windy gamma': AI(envs['windy'], gamma=0.9),  # note the change in value function and policy
        'too greedy': AI(envs['simple'], gamma=0), # unable to see only one step further, unable to reach target state
        'too windy': AI(envs['too windy'], gamma=0) # any action has random outcome, unable to learn anything
    }

    for name,ai in ais.items():
        print('Training AI: ' + name)
        ai.learn()
        print('Value function:')
        ai.print_value()
        print('Optimal policy:')
        ai.print_policy()
        print()
