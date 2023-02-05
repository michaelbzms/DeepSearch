from dataclasses import dataclass
from typing import Iterable
import torch
import numpy as np

from deep_search.search.action import Action
from deep_search.search.state import GameState


@dataclass
class ConnectFourAction(Action):
    row: int
    col: int
    player: int


class ConnectFourState(GameState):
    """
    Implements the Connect-Four game board as a nrows x ncols x 2 binary matrix
    """

    # class-wide params
    nrows: int = 6
    ncols: int = 7
    connect_num: int = 4

    def __init__(self, board: np.ndarray or None = None, turn: int = 1):
        if board is not None:
            self.board = board
            if board.shape != (self.nrows, self.ncols, 2):
                raise ValueError
        else:
            self.board = np.zeros((self.nrows, self.ncols, 2), dtype=np.int8)
        # the row that is next for play on each column
        self.top = self.board.sum(axis=-1).sum(axis=-1)   # sum the last two axis
        # player turn
        self.turn = turn
        # determine if there is a winner
        self.winner = self._calc_winner()

    def _calc_winner(self) -> int:
        # TODO: test
        for player_no in (1, 2):
            for j in range(self.ncols):
                for i in range(self.top[j]):
                    # horizontal
                    space_right = j <= self.ncols - self.connect_num
                    if space_right and self.board[i, j: j + self.connect_num, player_no - 1].sum() == self.connect_num:
                        return player_no
                    # vertical
                    space_up = i <= self.nrows - self.connect_num
                    if space_up and self.top[j] - i >= self.connect_num and self.board[i: i + self.connect_num, j, player_no - 1].sum() == self.connect_num:
                        return player_no
                    # diagonal
                    if space_right and space_up and (
                        self.board[i: i + self.connect_num, j: j + self.connect_num, player_no - 1].trace() == self.connect_num
                        or
                        np.fliplr(self.board[i: i + self.connect_num, j: j + self.connect_num, player_no - 1]).trace() == self.connect_num
                    ):
                        return player_no
        return 0

    def get_possible_actions(self) -> Iterable[ConnectFourAction]:
        return [
            ConnectFourAction(row=self.top[i], col=i, player=self.turn)
            for i in range(self.ncols)
            if self.top[i] < self.nrows
        ]

    def get_next_state(self, action: ConnectFourAction) -> GameState:
        x, y = action.row, action.col
        if self.board[x, y].sum() != 0:
            raise ValueError('Illegal action')
        new_board = np.copy(self.board)
        new_board[x, y, action.player - 1] = 1
        return ConnectFourState(board=new_board, turn=self._toggle_turn(self.turn))

    def is_final(self) -> bool:
        return self.get_winner() != 0

    def get_player_turn(self) -> int:
        return self.turn

    def get_winner(self) -> int:
        return self.winner

    def _toggle_turn(self, turn: int):
        return 2 if turn == 1 else 1

    def get_representation(self) -> torch.Tensor:
        """ Use the 3d board itself with players being different channels. """
        # For 2d version: return torch.FloatTensor(np.vstack((self.board[:, :, 0], self.board[:, :, 1]))
        return torch.FloatTensor(self.board)
