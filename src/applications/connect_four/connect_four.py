from dataclasses import dataclass
from typing import Iterable
import torch
import numpy as np
from numba import jit

from deep_search.search.action import Action
from deep_search.search.state import GameState


@dataclass
class ConnectFourAction(Action):
    col: int
    player: int

    def __str__(self):
        return f'{self.col+1}'


@jit(nopython=True)
def calc_winner_numba(nrows: int, ncols: int, connect_num: int, board: np.ndarray, top: np.ndarray) -> int:
    for player_no in range(1, 3):
        for j in range(ncols):
            for i in range(top[j]):
                # horizontal
                space_right = j <= ncols - connect_num
                if space_right and board[i, j: j + connect_num, player_no - 1].sum() == connect_num:
                    return player_no
                # vertical
                space_up = i <= nrows - connect_num
                if space_up and top[j] - i >= connect_num and board[i: i + connect_num, j, player_no - 1].sum() == connect_num:
                    return player_no
                # diagonal
                if space_right and space_up and (
                        (board[i: i + connect_num, j: j + connect_num, player_no - 1] * np.eye(4)).sum() == connect_num
                        or
                        (np.fliplr(board[i: i + connect_num, j: j + connect_num, player_no - 1]) * np.eye(4)).sum() == connect_num):
                    return player_no
    return 0   # no winner yet or draw


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
        self.top = self.board.sum(axis=-1).sum(axis=0)
        # player turn
        self.turn = turn
        # determine if there is a winner
        self.winner = self._calc_winner()

    def _calc_winner(self) -> int:
        return calc_winner_numba(self.nrows, self.ncols, self.connect_num, self.board, self.top)

    def get_possible_actions(self) -> Iterable[ConnectFourAction]:
        for i in range(self.ncols):
            if self.top[i] < self.nrows:
                yield ConnectFourAction(col=i, player=self.turn)

    def get_next_state(self, action: ConnectFourAction) -> GameState:
        x, y = self.top[action.col], action.col
        if self.board[x, y].sum() != 0:
            raise ValueError('Illegal action')
        new_board = np.copy(self.board)
        new_board[x, y, action.player - 1] = 1
        return ConnectFourState(board=new_board, turn=self._toggle_turn(self.turn))

    def is_final(self) -> bool:
        if self.get_winner() != 0:
            return True
        return np.all(self.board.sum(axis=-1) != 0)  # no empty squares

    def get_player_turn(self) -> int:
        return self.turn

    def get_winner(self) -> int:
        return self.winner

    def _toggle_turn(self, turn: int):
        return 2 if turn == 1 else 1

    def get_representation(self) -> torch.Tensor:
        """ Use the 3d board itself with players being different channels. """
        # For 2d version: return torch.FloatTensor(np.vstack((self.board[:, :, 0], self.board[:, :, 1]))
        return torch.FloatTensor(self.board).permute(2, 0, 1)

    def __str__(self) -> str:
        return str(self.board[::-1, :, 0] + 2 * self.board[::-1, :, 1])

    def __eq__(self, other):
        return self.__id() == other.__id()

    def __id(self):
        return np.packbits(self.board).tobytes()

    def __hash__(self) -> int:
        return hash(self.__id())

    def draw(self) -> None:
        import pygame
        # Incorporated hastily from: https://www.askpython.com/python/examples/connect-four-game
        pygame.init()
        # define colors
        BLUE = (4, 55, 225)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        YELLOW = (255, 255, 0)
        # define our screen size
        SQUARESIZE = 100
        # define width and height of board
        width = self.ncols * SQUARESIZE
        height = (self.nrows + 1) * SQUARESIZE
        size = (width, height)
        RADIUS = int(SQUARESIZE / 2 - 5)
        screen = pygame.display.set_mode(size)
        for c in range(self.ncols):
            for r in range(self.nrows):
                pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
                pygame.draw.circle(screen, BLACK, (int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
        for c in range(self.ncols):
            for r in range(self.nrows):
                if self.board[r, c, 0] == 1:
                    pygame.draw.circle(screen, RED, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
                elif self.board[r, c, 1] == 1:
                    pygame.draw.circle(screen, YELLOW, (int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
        pygame.display.update()

    def serialize(self) -> np.array:
        """ serialize state info TODO: check """
        return self.board.flatten()

    @staticmethod
    def calculate_turn(board: np.ndarray):
        return 1 if board.sum() % 2 == 0 else 2  # TODO: calculate based on board

    @staticmethod
    def deserialize(serial: np.ndarray) -> GameState:
        board = ConnectFourState.deserialize_board(serial)
        return ConnectFourState(
            board=board,
            turn=ConnectFourState.calculate_turn(board)
        )

    @staticmethod
    def deserialize_board(serial: np.ndarray) -> np.ndarray:
        return serial.reshape((ConnectFourState.nrows, ConnectFourState.ncols, 2))
