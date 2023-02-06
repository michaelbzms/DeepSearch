import numpy as np
from numba import jit

from applications.connect_four.connect_four import ConnectFourState


@jit(nopython=True)
def max_consecutive_squares(board: np.ndarray, player_no: int) -> float:
    max_cons: int = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # max consecutive horizontal
            k = 0
            while j + k < board.shape[1] and board[i, j + k, player_no] == 1:
                k += 1
            max_cons = max(max_cons, k)
            # vertical
            k = 0
            while i + k < board.shape[0] and board[i + k, j, player_no] == 1:
                k += 1
            max_cons = max(max_cons, k)
            # diagonal
            k = 0
            while j + k < board.shape[1] and i + k < board.shape[0] and board[i + k, j + k, player_no] == 1:
                k += 1
            max_cons = max(max_cons, k)
            k = 0
            while j - k >= 0 and i + k < board.shape[0] and board[i + k, j - k, player_no] == 1:
                k += 1
            max_cons = max(max_cons, k)
    return max_cons


@jit(nopython=True)
def total_consecutive_squares(board: np.ndarray, player_no: int) -> float:
    """
    Only 2+ consecutive count and more consecutive count for more.
    """
    total_cons: int = 0
    scale: int = 2
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # horizontal
            k = 0
            while j + k < board.shape[1] and board[i, j + k, player_no] == 1:
                k += 1
            if k > 1:
                total_cons += k ** scale
            # vertical
            k = 0
            while i + k < board.shape[0] and board[i + k, j, player_no] == 1:
                k += 1
            if k > 1:
                total_cons += k ** scale
            # diagonal
            k = 0
            while j + k < board.shape[1] and i + k < board.shape[0] and board[i + k, j + k, player_no] == 1:
                k += 1
            if k > 1:
                total_cons += k ** scale
            k = 0
            while j - k >= 0 and i + k < board.shape[0] and board[i + k, j - k, player_no] == 1:
                k += 1
            if k > 1:
                total_cons += k ** scale
    return total_cons


def max_consecutive_squares_eval(s: ConnectFourState) -> float:
    """
    Return the difference between our max consecutive squares and theirs as an eval.
    """
    turn_bonus = 1.0 if s.turn == 1 else -1.0
    max_cons = [0, 0]
    for player_no in range(2):
        max_cons[player_no] = max_consecutive_squares(s.board, player_no)
    return (max_cons[0] - max_cons[1] + turn_bonus) / 16  # so that it is in [-1, 1]


def total_consecutive_squares_eval(s: ConnectFourState) -> float:
    """
    Return the difference between our total consecutive squares and theirs but squeshed in [-1, 1] as an eval.
    """
    turn_bonus = 1.0 if s.turn == 1 else -1.0
    total_cons = [0, 0]
    for player_no in range(2):
        total_cons[player_no] = total_consecutive_squares(s.board, player_no)
    return np.tanh((total_cons[0] - total_cons[1] + turn_bonus) * 0.1)    # so that it is in [-1, 1]
