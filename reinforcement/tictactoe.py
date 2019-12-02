import numpy as np

class Environment:
    def __init__(self, length=3):
        self.length = length
        self.range = range(self.length)
        self.roles = (1, -1)
        self.symbol_map = {0:' ', 1:'x', -1:'o'}
        self.n_states = len(self.symbol_map) ** (self.length * self.length)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.length, self.length))
        self.winner = None

    def print_board(self):
        print('    ', end='')
        print(*[str(i + 1) for i in range(self.length)], sep = '   ')
        for i in range(self.length):
            print('  -' + self.length * '----')
            print(str(i + 1) + ' |', end='')
            for j in range(self.length):
                print(' ' + self.symbol_map[self.board[i,j]] + ' |', end='')
            print('')
        print('  -' + self.length * '----')

    # Get the list of empty cells
    def get_valid_moves(self):
        result = []
        for i in self.range:
            for j in self.range:
                if self.board[i,j] == 0:
                    result.append((i, j))
        return result

    # Check that the mvoe is valid, before applying it
    def apply_move(self, move, role):
        assert isinstance(move, tuple)
        i, j = move
        if (i in self.range and j in self.range and self.board[i,j] == 0):
            self.board[i,j] = role
            return True
        else:
            return False

    # Return a board that is a copy of the environment board,
    # with one additional move
    def simulate_move(self, move, role):
        assert isinstance(move, tuple)
        i, j = move
        assert (i in self.range and j in self.range and self.board[i,j] == 0)
        board = self.board.copy()
        board[i,j] = role
        return board

    def is_gameover(self):
        # Check if empty
        if np.all(self.board == 0):
            return False
        # Loop for each role
        for role in self.roles:
            three = role * self.length
            # Check rows and cols
            for i in self.range:
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
    # In this mode, we are not letting AI players learning from the game.
    # We would need to add observe() after move().
    def play_game(self, player1, player2, learn=False):
        self.reset()
        assert isinstance(player1, Player) and isinstance(player2, Player)
        players = dict(zip(self.roles, (player1, player2)))
        # Draw board
        print(player1.name + ' against ' + player2.name)
        self.print_board()
        # Continue until gameover
        current_role = self.roles[0]  # =1
        while True:
            # Current player makes the move. Repeat until valid move
            players[current_role].move(env=self, role=current_role)
            # Draw board
            print(players[current_role].name + ' moves:')
            self.print_board()
            # Check if gameover
            if self.is_gameover():
                if learn:
                    player0.learn(winner=self.winner)
                    player1.learn(winner=self.winner)
                if self.winner == 0:
                    print('It\'s a draw!')
                else:
                    print('Player ' + players[self.winner].name + ' (' \
                        + self.symbol_map[self.winner] + ') wins!')
                break
            # Switch players
            current_role *= -1

    # This functions facilitate training the AI, by making the same agent
    # playing against itself multiple times.
    def train_AI(self, ai, n=10000):
        assert isinstance(ai, AI)
        # Train n times
        print('Training progress:')
        for i in range(n):
            # Print simulation progress
            if (i + 1) % (n // 10) == 0: print(str(100 * (i + 1) // n) + '%')
            self.reset()
            current_role = self.roles[0]  # =1
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
        self.eps = min(1, max(0, eps))
        self.alpha = min(1, max(0, alpha))
        self.history = []
        self.init_weights(env)

    def init_weights(self, env):
        # Initialize states randomly between -0.1 and 0.1
        self.V = np.random.random(env.n_states) / 5. - 0.1
        # Initialize end state value to winning role (0 if draw)
        for state, winner in self.generate_endgames(env):
            self.V[state] = winner
        env.reset()

    # Recursive function that generates all possible final states.
    # Result is a list of tuples, with status and winner of each endgame.
    # Note that we do not check if states are valid (it does not matter).
    def generate_endgames(self, env, i=0, j=0):
        endgames = []
        #for symbol in env.symbol_map.keys():  # include endgames with empty cells
        for symbol in env.roles:  # ignore endgames with empty cells. It works better. Why?!
            env.board[i,j] = symbol
            if j == env.length-1:
                if i == env.length-1:
                    # Board complete, evaluate final state
                    if env.is_gameover():
                        endgame = (self.get_state(env), env.winner)
                        endgames.append(endgame)
                else:
                    # Next row: increase i, reset j
                    endgames += self.generate_endgames(env, i+1, j=0)
            else:
                # Next column: keep i, increase j
                endgames += self.generate_endgames(env, i, j+1)
        return endgames

    # Get the unique integer representation of board state.
    # Uniqueness ensured by number of values.
    # Used to identify states in the weights array.
    def get_state(self, env, board=None):
        if board is None: board = env.board
        base = len(env.symbol_map)
        k = 0  # base exponent (represents the cell)
        value_map = dict(zip(env.symbol_map.keys(), range(len(env.symbol_map))))  # cell values encoding
        result = 0
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                result += value_map[board[i,j]] * (base ** k)
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
            # We multiply by role, because if role==-1 we want to maximize -values.
            values = role * np.array([
                self.V[self.get_state(env, board=env.simulate_move(move, role))]
                for move in valid_moves])
            move = valid_moves[values.argmax()]
        # Make the move
        assert env.apply_move(move, role)

    def observe(self, env):
        self.history.append(self.get_state(env))

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
        if V.shape == (env.n_states,):
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

    # Try importing pre-trained weights, otherwise retrain.
    EPOCHS = 10000
    try:
        filepath = 'models/tictactoe_' + str(EPOCHS) + '.out'
        ai.set_weights(np.loadtxt(filepath), env)
        print('AI weights restored from ' + filepath)
    except IOError:
        print(filepath + ' not found, AI will be trained.')
        # Train AI by making it play multiple times against himself
        env.train_AI(ai, n=EPOCHS)
        # Save weights to text file for future use
        np.savetxt(filepath, ai.get_weights())

    # Play games between different players.
    tournament = [
        (ai, mon),  # AI against Monkey
        (ai, ai),   # AI against itself
        (ai, hum)   # AI against Human
    ]
    for players in tournament:
        while True:
            env.play_game(*players)
            players = players[::-1]  # Reverse board
            answer = input("Play again? [Y/n]: ")
            if answer and answer.lower()[0] == 'n':
                break
