from abc import ABC, abstractmethod
from typing import Callable

from deep_search.search.action import Action
from deep_search.search.algorithms import alphabeta, GameNode
from deep_search.search.state import State, GameState


class Agent(ABC):
    @abstractmethod
    def decide_action(self, state: State) -> Action:
        raise NotImplemented


class GameAgent(Agent):
    @abstractmethod
    def decide_action(self, state: GameState) -> Action:
        raise NotImplemented


class AlphaBetaAgent(GameAgent):
    def __init__(self, depth: int, player: int, heuristic: Callable[[GameState], float], verbose=True):
        self.depth = depth
        self.player = 'max' if player == 1 else 'min'
        self.heuristic = heuristic
        self.verbose = verbose

    def decide_action(self, state: GameState) -> Action:
        # run alpha-beta from root node state
        minimax_value, actions = alphabeta(
            node=GameNode(state),
            depth=self.depth,
            player=self.player,
            heuristic=self.heuristic
        )
        if self.verbose:
            print(f'Player {self.player} calculated minimax value: {minimax_value}')
        # return first action of principal variation (i.e. what we should do under perfect play)
        return actions[0]
