import numpy as np

class Environment:
    def __init__(self):
        self.LENGTH = 3
        self.RANGE = range(self.LENGTH)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.LENGTH, self.LENGTH))
        self.winner = None

    def print(self):
        print('    1   2   3')
        for i in range(self.LENGTH):
            print('  -------------')
            print(str(i + 1) + ' |', end='')
            for j in range(self.LENGTH):
                print(' ', end='')
                if self.board[i,j] == 1:
                    print('x |', end='')
                elif self.board[i,j] == -1:
                    print('o |', end='')
                else:
                    print('  |', end='')
            print('')
        print('  -------------')

    # Get the list of empty cells
    def get_valid_moves(self):
        result = []
        for i in self.RANGE:
            for j in self.RANGE:
                if self.board[i,j] == 0:
                    result.append((i, j))
        return result

    # Check that the mvoe is valid, before applying it
    def apply_move(self, move, role):
        assert isinstance(move, tuple)
        i, j = move
        if (i in self.RANGE and j in self.RANGE and self.board[i,j] == 0):
            self.board[i,j] = role
            return True
        else:
            return False

    # Return a board that is a copy of the environment board,
    # with one additional move
    def simulate_move(self, move, role):
        assert isinstance(move, tuple)
        i, j = move
        assert (i in self.RANGE and j in self.RANGE and self.board[i,j] == 0)
        board = self.board.copy()
        board[i,j] = role
        return board

    def is_gameover(self):
        # Check if empty
        if np.all(self.board == 0):
            return False
        # Loop for each role
        for role in (-1, 1):
            three = role * self.LENGTH
            # Check rows and cols
            for i in self.RANGE:
                if self.board[i,:].sum() == three \
                or self.board[:,i].sum() == three:
                    self.winner = role
                    return True
            # Check diags
            if self.board.trace() == three or self.board[::-1].trace() == three:
                self.winner = role
                return True
        # Check for a draw
        if np.all(self.board != 0):
            self.winner = 0
            return True
        # Default
        return False

    # Start a game where players make moves in turns.
    # If learn is True, AI players update the weights based on the game history.
    def play_game(self, player0, player1, learn=False):
        self.reset()
        assert isinstance(player0, Player) and isinstance(player1, Player)
        players = (player0, player1)
        roles = (1, -1)
        symbols = ('x', 'o')
        current_player = 0
        # Draw board
        print(player0.name + ' against ' + player1.name)
        self.print()
        # Continue until gameover
        while True:
            # Current player makes the move. Repeat until valid move
            players[current_player].move(env=self, role=roles[current_player])
            # Draw board
            print(players[current_player].name + ' moves:')
            self.print()
            # Check if gameover
            if self.is_gameover():
                if learn:
                    player0.learn(winner=self.winner)
                    player1.learn(winner=self.winner)
                if self.winner == 0:
                    print('It\'s a draw!')
                else:
                    print('Player ' + players[current_player].name + ' (' \
                        + symbols[current_player] + ') wins!')
                break
            # Switch players
            current_player = 1 - current_player

    # This functions facilitate training the AI, by making the same agent
    # playing against itself.
    def train_AI(self, ai, n=10000):
        assert isinstance(ai, AI)
        # Train n times
        for i in range(n):
            # Print simulation progress
            if (i + 1) % (n // 100) == 0:
                print("Training progress: " + str(100*(i+1)//n) + "%")
            self.reset()
            current_role = 1
            # Continue until gameover
            while True:
                # AI makes the move
                ai.move(env, training_mode=True, role=current_role)
                # AI updates its history
                ai.observe(env)
                # Check if gameover
                if self.is_gameover():
                    ai.learn(winner=self.winner)
                    break
                # Switch side (only affect the role here)
                current_role *= -1


class Player:
    def __init__(self):
        self.name = 'Player'

    def learn(self):
        pass


class Human(Player):
    def __init__(self):
        self.name = 'Human'

    def move(self, env, role):
        # Repeat until valid move
        while True:
            # Repeat until valid input
            while True:
                answer = input('Enter coordinates i,j for your next move (i,j=1..3): ').split(',')
                if len(answer) == 2:
                  try:
                    move = (int(answer[0]) - 1, int(answer[1]) - 1)
                    break
                  except ValueError:
                    pass
            # Try to update the board
            if env.apply_move(move, role):
                break
            else:
                print('Invalid move.')


class Monkey(Player):
    def __init__(self):
        self.name = 'Monkey'

    def move(self, env, role):
        valid_moves = env.get_valid_moves()
        move = valid_moves[np.random.choice(len(valid_moves))]
        assert env.apply_move(move, role)


class AI(Player):
    def __init__(self, env, alpha=0.5, eps=0.1):
        self.name = 'AI'
        self.eps = eps
        self.alpha = alpha
        self.history = []
        self.init_weights(env)

    def init_weights(self, env):
        n_states = 3**(env.LENGTH*env.LENGTH)
        # Initialize all states to 0.5
        self.V = np.full(n_states, 0.5)
        # Initialize end state value to winning role (0 if draw)
        for state, winner in self.generate_endgames(env):
            self.V[state] = winner
        env.reset()

    # Recursive function to generate all possible final states.
    # Result is a list of tuples, with status and winner of each final state.
    # Used to set initial weights.
    def generate_endgames(self, env, i=0, j=0):
        endgames = []
        for role in (-1, 1):
            env.board[i,j] = role
            if j == env.LENGTH-1:
                if i == env.LENGTH-1:
                    # Board complete, evaluate final state
                    assert env.is_gameover()
                    endgame = (self.get_state(env.board), env.winner)
                    endgames.append(endgame)
                else:
                    # Next row: increase i, reset j
                    endgames += self.generate_endgames(env, i+1, j=0)
            else:
                # Next column: keep i, increase j
                endgames += self.generate_endgames(env, i, j+1)
        return endgames

    # Get the integer representation of board state (a base 3 number).
    # Used to identify states in the weights array.
    def get_state(self, board):
        result = 0
        value_map = {0:0, -1:1, 1:2}  # base 3 encoding (for values in a cell)
        k = 0  # base 3 exponent (different for each cell)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                result += value_map[board[i,j]] * (3 ** k)
                k += 1  # change cell
        return result

    # Make the best move based on max (or min) state value.
    # If in training mode, adopt epsilon-greedy exploration strategy.
    def move(self, env, role, training_mode=False):
        valid_moves = env.get_valid_moves()
        # Choose between random and best move with epsilon-probability
        r = np.random.rand()
        if training_mode and r < self.eps:
            move = valid_moves[np.random.choice(len(valid_moves))]
        else:
            # Calculate the values of states for all valid moves (max 9 states)
            values = np.array([self.V[self.get_state(env.simulate_move(move, role))] for move in valid_moves])
            # If role==1 we want to maximize, otherwise to minimize
            move = valid_moves[values.argmax() if role==1 else values.argmin()]
        # Make the move
        assert env.apply_move(move, role)

    def observe(self, env):
        self.history.append(self.get_state(env.board))

    # Recursively update the weights based on current game history and outcome.
    def learn(self, winner):
        # Last state value is equal to winning role (0 if draw)
        self.V[self.history[-1]] = winner
        next_value = winner
        # Recursively update the values of all states visited during the game
        for state in reversed(self.history[:-1]):
            self.V[state] = self.V[state] + self.alpha * (next_value - self.V[state])
            next_value = self.V[state]
        # Reset history
        self.history = []

    def get_weights(self):
        return self.V

    def set_weights(self, V, env):
        n_states = 3**(env.LENGTH*env.LENGTH)
        if V.shape == (n_states,):
            self.V = V
            return True
        else:
            return False

    def copy(self, name='AI'):
        clone = AI(self.alpha, self.eps);
        clone.V = self.V
        return clone


if __name__ == '__main__':

    env = Environment()
    hum = Human()
    mon = Monkey()
    ai = AI(env)  # the AI needs env to initialise proper weights

    # Train AI by making it play multiple times against himself
    env.train_AI(ai)
    # # Save weights to CSV/JSON for future use
    # V = ai.get_weights()

    tournament = [
        (ai, mon),  # AI against Monkey
        (ai, ai),   # AI against itself
        (ai, hum)  # AI against Human
    ]
    for players in tournament:
        while True:
            env.play_game(*players)
            players = players[::-1]  # Reverse board
            answer = input("Play again? [Y/n]: ")
            if answer and answer.lower()[0] == 'n':
                break
