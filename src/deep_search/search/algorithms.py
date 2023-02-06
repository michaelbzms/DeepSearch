import math
from typing import Literal, Callable

from deep_search.search.state import State, GameState


class Node:
    def __init__(self, state: State):
        self.state = state

    def successors(self) -> State:
        actions = self.state.get_possible_actions()
        for action in actions:
            next_state = self.state.get_next_state(action)
            yield GameNode(next_state)

    def is_final(self):
        return self.state.is_final()


class GameNode(Node):
    def __init__(self, state: GameState):
        super().__init__(state)
        self.state = state       # set to GameState now

    def reward(self) -> float:
        winner = self.state.get_winner()
        if winner is None:
            raise ValueError('Asked for reward of unfinished game')
        elif winner == 1:
            return +1.0
        elif winner == 2:
            return -1.0
        else:
            raise ValueError('Could not recognize the winner.')


toggle_minimax_player = lambda x: ('min' if x == 'max' else 'max')


def minimax(node: GameNode, depth: int, player: Literal['max', 'min'], heuristic: Callable[[GameState], float]):
    # if reach final state return actual value
    if node.is_final():
        return node.reward()
    # if reach end of simulation depth apply heuristic
    if depth == 0:
        return heuristic(node.state)
    # recursively build the minimax tree
    if player == 'max':
        max_value = -math.inf
        for succ in node.successors():
            max_value = max(max_value, minimax(succ, depth - 1, toggle_minimax_player(player), heuristic))
        return max_value
    else:
        min_value = math.inf
        for succ in node.successors():
            min_value = min(min_value, minimax(succ, depth - 1, toggle_minimax_player(player), heuristic))
        return min_value


def alphabeta(node: GameNode, depth: int, player: Literal['max', 'min'], heuristic: Callable[[GameState], float], a: float = -math.inf, b: float = math.inf):
    # if reach final state return actual value
    if node.is_final():
        return node.reward()
    # if reach end of simulation depth apply heuristic
    if depth == 0:
        return heuristic(node.state)
    # recursively build the minimax tree
    if player == 'max':
        max_value = -math.inf
        for succ in node.successors():
            max_value = max(max_value, alphabeta(succ, depth - 1, toggle_minimax_player(player), heuristic, a, b))
            a = max(a, max_value)      # fail-soft gives more info
            if max_value > b:          # better than the worst that min is guaranteed to be able to go for
                break                  # prune this node entirely
        return max_value
    else:
        min_value = math.inf
        for succ in node.successors():
            min_value = min(min_value, alphabeta(succ, depth - 1, toggle_minimax_player(player), heuristic, a, b))
            b = min(b, min_value)      # fail-soft gives more info
            if min_value < a:          # better than the worst that max is guaranteed to be able to go for
                break                  # prune this node entirely
        return min_value
