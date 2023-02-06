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
    """
    IMPORTANT: We only decrease the depth on MAX nodes, so that we always end the simulation on MIN nodes, i.e. we
    call the heuristic on MIN nodes, where it's the MIN player's turn to play. This is so that we optimize learning
    to evaluate a position after we have played a move rather than before playing it, assuming we are the MAX player.
    """
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
            value = minimax(succ, depth - 1, toggle_minimax_player(player), heuristic)
            max_value = max(max_value, value)
        return max_value
    else:
        min_value = math.inf
        for succ in node.successors():
            value = minimax(succ, depth, toggle_minimax_player(player), heuristic)
            min_value = min(min_value, value)
        return min_value


def alphabeta(node: GameNode, depth: int, player: Literal['max', 'min'], heuristic: Callable[[GameState], float], a: float = -math.inf, b: float = math.inf):
    """
    IMPORTANT: We only decrease the depth on MAX nodes, so that we always end the simulation on MIN nodes, i.e. we
    call the heuristic on MIN nodes, where it's the MIN player's turn to play. This is so that we optimize learning
    to evaluate a position after we have played a move rather than before playing it, assuming we are the MAX player.
    """
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
            value = alphabeta(succ, depth - 1, toggle_minimax_player(player), heuristic, a, b)
            max_value = max(max_value, value)
            a = max(a, max_value)      # fail-soft gives more info
            if max_value > b:          # better than the worst that min is guaranteed to be able to go for
                break                  # prune this node entirely
        return max_value
    else:
        min_value = math.inf
        for succ in node.successors():
            value = alphabeta(succ, depth, toggle_minimax_player(player), heuristic, a, b)
            min_value = min(min_value, value)
            b = min(b, min_value)      # fail-soft gives more info
            if min_value < a:          # better than the worst that max is guaranteed to be able to go for
                break                  # prune this node entirely
        return min_value
