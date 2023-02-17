import math
from typing import Literal, Callable

from deep_search.search.action import Action
from deep_search.search.state import State, GameState


class Node:
    def __init__(self, state: State):
        self.state = state

    def successors(self) -> State:
        actions = self.state.get_possible_actions()
        for action in actions:
            next_state = self.state.get_next_state(action)
            yield action, GameNode(next_state)

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
            return 0.0
        elif winner == 0:   # draw
            return 0.5
        else:
            raise ValueError('Could not recognize the winner.')


toggle_minimax_player = lambda x: ('min' if x == 'max' else 'max')


def minimax(node: GameNode, depth: int, player: Literal['max', 'min'], heuristic: Callable[[GameState], float],
            transposition_table: dict[GameState, float] or None = None) -> (float, [Action]):
    """
    IMPORTANT: We only decrease the depth on MAX nodes, so that we always end the simulation on MIN nodes, i.e. we
    call the heuristic on MIN nodes, where it's the MIN player's turn to play. This is so that we optimize learning
    to evaluate a position after we have played a move rather than before playing it, assuming we are the MAX player.
    """
    # if reach final state return actual value
    if node.is_final():
        return node.reward(), []
    # if reach end of simulation depth apply heuristic
    if depth == 0:
        return heuristic(node.state), []
    # recursively build the minimax tree
    if player == 'max':
        max_value = -math.inf
        max_actions = None
        for action, succ in node.successors():
            if transposition_table is not None and succ.state in transposition_table:
                value, actions = transposition_table[succ.state]
            else:
                value, actions = minimax(succ, depth - 1, toggle_minimax_player(player), heuristic, transposition_table)
                if transposition_table is not None:
                    transposition_table[succ.state] = (value, actions)
            if value > max_value:
                max_value = value
                max_actions = [action] + actions
        return max_value, max_actions
    else:
        min_value = math.inf
        min_actions = None
        for action, succ in node.successors():
            if transposition_table is not None and succ.state in transposition_table:
                value, actions = transposition_table[succ.state]
            else:
                value, actions = minimax(succ, depth, toggle_minimax_player(player), heuristic, transposition_table)
                if transposition_table is not None:
                    transposition_table[succ.state] = (value, actions)
            if value < min_value:
                min_value = value
                min_actions = [action] + actions
        return min_value, min_actions


def alphabeta(node: GameNode, depth: int, player: Literal['max', 'min'], heuristic: Callable[[GameState], float],
              transposition_table: dict[GameState, [float, float, float]] or None = None,
              a: float = -math.inf, b: float = math.inf) -> (float, [Action]):
    """
    IMPORTANT: We only decrease the depth on MAX nodes, so that we always end the simulation on MIN nodes, i.e. we
    call the heuristic on MIN nodes, where it's the MIN player's turn to play. This is so that we optimize learning
    to evaluate a position after we have played a move rather than before playing it, assuming we are the MAX player.

    TODO:
        Using a Transposition Table (TT) in Alpha-Beta pruning is a lot more challenging due to the cut-offs.
        It's very much possible that this does not work correctly in all cases so avoid using a TT here without making sure.
        Intuition of why I think it might be correct:
            Assuming that we use a new TT every time we call alphabeta on a root node (!) then the DFS can only make the (a, b)
            bounds tighter as we go. Therefore, if we store some (a, b) pair in the TT then this cannot have occurred from
            less tight (a, b) pairs in the past, hence it is safe to use the (a, b) pairs that were stored as those cannot be worse
            than the result of doing the call again anyway.
    """
    # if reach final state return actual value
    if node.is_final():
        return node.reward(), []
    # if reach end of simulation depth apply heuristic
    if depth == 0:
        return heuristic(node.state), []
    # recursively build the minimax tree
    if player == 'max':
        max_value = -math.inf
        max_actions = None
        for action, succ in node.successors():
            if transposition_table is not None and succ.state in transposition_table:
                value, _a, _b, actions = transposition_table[succ.state]
                a = max(a, _a)
                b = min(b, _b)
            else:
                value, actions = alphabeta(succ, depth - 1, toggle_minimax_player(player), heuristic, transposition_table, a, b)
                if transposition_table is not None:
                    transposition_table[succ.state] = (value, a, b, actions)
            if value > max_value:
                max_value = value
                max_actions = [action] + actions
            a = max(a, max_value)      # fail-soft gives more info
            if max_value > b:          # better than the worst that min is guaranteed to be able to go for
                break                  # prune this node entirely
        return max_value, max_actions
    else:
        min_value = math.inf
        min_actions = None
        for action, succ in node.successors():
            if transposition_table is not None and succ.state in transposition_table:
                value, _a, _b, actions = transposition_table[succ.state]
                a = max(a, _a)
                b = min(b, _b)
            else:
                value, actions = alphabeta(succ, depth, toggle_minimax_player(player), heuristic, transposition_table, a, b)
                if transposition_table is not None:
                    transposition_table[succ.state] = (value, a, b, actions)
            if value < min_value:
                min_value = value
                min_actions = [action] + actions
            b = min(b, min_value)      # fail-soft gives more info
            if min_value < a:          # better than the worst that max is guaranteed to be able to go for
                break                  # prune this node entirely
        return min_value, min_actions
