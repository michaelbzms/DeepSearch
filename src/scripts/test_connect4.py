import numpy as np
import time

from applications.connect_four.connect_four import ConnectFourState, ConnectFourAction
from deep_search.search.algorithms import GameNode, minimax, alphabeta


if __name__ == '__main__':

    # start state
    s = ConnectFourState()

    # determine some actions
    actions = [
        ConnectFourAction(col=2, player=1),
        ConnectFourAction(col=3, player=2),
        ConnectFourAction(col=3, player=1),
        ConnectFourAction(col=4, player=2),
        ConnectFourAction(col=1, player=1),
        ConnectFourAction(col=5, player=2),
    ]

    # play actions
    for a in actions:
        s = s.get_next_state(a)
    print(s)

    start_node = GameNode(s)
    depth = 3


    def simple_connect4_heuristic(s: ConnectFourState) -> float:
        """
        Return the difference between our max consecutive squares and theirs as an eval.
        TODO: Speed-up?
        TODO: Heuristic should take into account whose turn it is to play.. here we assume it is MIN's.
        """
        max_cons = np.array([0, 0])
        for player_no in range(2):
            for i in range(s.board.shape[0]):
                for j in range(s.board.shape[1]):
                    # max consecutive horizontal
                    k = 0
                    while j + k < s.board.shape[1] and s.board[i, j + k, player_no] == 1:
                        k += 1
                    max_cons[player_no] = max(max_cons[player_no], k)
                    # vertical
                    k = 0
                    while i + k < s.board.shape[0] and s.board[i + k, j, player_no] == 1:
                        k += 1
                    max_cons[player_no] = max(max_cons[player_no], k)
                    # diagonal
                    k = 0
                    while j + k < s.board.shape[1] and i + k < s.board.shape[0] and s.board[i + k, j + k, player_no] == 1:
                        k += 1
                    max_cons[player_no] = max(max_cons[player_no], k)
                    k = 0
                    while j - k >= 0 and i + k < s.board.shape[0] and s.board[i + k, j - k, player_no] == 1:
                        k += 1
                    max_cons[player_no] = max(max_cons[player_no], k)
        return max_cons[0] - max_cons[1]

    print('Heuristic:', simple_connect4_heuristic(start_node.state))

    start = time.time()
    value = minimax(start_node, depth=2, player=1, heuristic=simple_connect4_heuristic)
    end = time.time()
    print(f'Minimax: {value}, time = {end - start} sec.')

    start = time.time()
    value = alphabeta(start_node, depth=2, player=1, heuristic=simple_connect4_heuristic)
    end = time.time()
    print(f'Alpha-beta: {value}, time = {end - start} sec.')
