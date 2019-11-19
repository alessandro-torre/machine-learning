import numpy as np

class Environment:
    def __init__(self):
        self.LENGTH = 3
        self.range = range(self.LENGTH)
        self.board = np.zeros((self.LENGTH, self.LENGTH))
        self.symbols = (-1, 1)
        self.gameover = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((self.LENGTH, self.LENGTH))

    def draw_board(self):
        print('    1   2   3')
        for i in range(self.LENGTH):
            print('  -------------')
            print(str(i + 1) + ' |', end='')
            for j in range(self.LENGTH):
                print(' ', end='')
                if self.board[i,j] == self.symbols[True]:
                    print('x |', end='')
                elif self.board[i,j] == self.symbols[False]:
                    print('o |', end='')
                else:
                    print('  |', end='')
            print('')
        print('  -------------')

    def is_valid(self, i, j):
        return (i in self.range and j in self.range and self.board[i, j] == 0)

    def set(self, i, j, symbol):
        assert symbol in self.symbols
        if self.is_valid(i, j):
            self.board[i, j] = symbol
            return True
        else:
            return False

    def check_gameover(self):
        # Check if empty
        if np.all(self.board == 0):
            return False
        # Loop for each player
        for symbol in self.symbols:
            three = symbol * self.LENGTH
            # Check rows and cols
            for i in self.range:
                if self.board[i,:].sum() == three \
                or self.board[:,i].sum() == three:
                    self.winner = symbol
                    self.gameover = True
                    return self.gameover
            # Check diags
            if self.board.trace() == three or self.board[::-1].trace() == three:
                self.winner = symbol
                self.gameover = True
                return self.gameover
        # Check for a draw
        if np.all(self.board != 0):
            self.gameover = True
            return self.gameover

    def play_game(self, player1, player2):
        self.reset()
        assert isinstance(player1, Player) and isinstance(player2, Player)
        player1.name, player2.name = ('Player 1', 'Player 2')
        player1.first, player2.first = True, False
        player1.symbol = self.symbols[player1.first]
        player2.symbol = self.symbols[player2.first]
        current_player = None
        self.draw_board()
        # Continue until gameover
        while True:
            # Switch players
            if current_player == player1:
                current_player = player2
            else:
                current_player = player1
            # Current player makes the move. Repeat until valid move
            while True:
                i, j = current_player.move(self)
                if env.set(i, j, current_player.symbol):
                    break
                else:
                    print('Invalid move.')
            # Draw board
            self.draw_board()
            # Check if gameover
            if self.check_gameover():
                if self.winner is None:
                    print('It\'s a draw!')
                else:
                    print(current_player.name + ' wins!')
                break

    def train_AI(self, ai, n=1000):
        assert isinstance(ai, AI)
        # Train n times
        for i in range(n):
            self.reset()
            ai.first = True
            ai.symbol = self.symbols[ai.first]
            # Continue until gameover
            while True:
                i, j = ai.move(env)
                assert env.set(i, j, ai.symbol)
                # Change playing side
                ai.first = not ai.first
                ai.symbol = self.symbols[ai.first]
                # Check if gameover
                if self.check_gameover(): break


class Player:
    def __init__(self):
        self.symbol = None
        self.name   = None
        self.first  = None


class Human(Player):
    def move(self, env):
        answer = []
        while len(answer) != 2:
            answer = input('Enter coordinates i,j for your next move (i,j=1..3): ').split(',')
        return (int(answer[0]) - 1, int(answer[1]) - 1)


class Monkey(Player):
    def move(self, env):
        while True:
            i = np.random.choice(env.LENGTH)
            j = np.random.choice(env.LENGTH)
            if env.is_valid(i, j): return (i, j)


# Should we train Ai both to start 1st and 2nd?
# Can we train the same AI for this? Two different value functions?
class AI(Player):
    def __init__(self, alpha=0.5, eps=0.1):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.init_weights()

    def init_weights(self):
        pass

    def move(self, env):
        # Select the proper weights
        if self.first:
            V = V.first
        else:
            V = V.second
        pass

    def save_weights(self):
        # save value state function to JSON/CSV
        pass

    def load_weghts(self):
        # load saved value state function
        pass


if __name__ == '__main__':

    h = Human()
    m = Monkey()
    ai = AI()
    env = Environment()

    # # Train Agent by making him play multiple times against himself
    # env.train_AI(ai, n=1000)

    # Human against Monkey
    while True:
        env.play_game(m, h)
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break

    # # Human against AI
    # while True:
    #     env.play_game(m, h)
    #     answer = input("Play again? [Y/n]: ")
    #     if answer and answer.lower()[0] == 'n':
    #         break
    #
    # # Monkey against AI
    # while True:
    #     env.play_game(m, h)
    #     answer = input("Play again? [Y/n]: ")
    #     if answer and answer.lower()[0] == 'n':
    #         break
