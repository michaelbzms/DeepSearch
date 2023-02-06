import time

from applications.connect_four.connect_four import ConnectFourState, ConnectFourAction
from applications.connect_four.heuristics import max_consecutive_squares_eval
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
        ConnectFourAction(col=4, player=2),
        ConnectFourAction(col=5, player=1),
        ConnectFourAction(col=5, player=2),
    ]

    # play actions
    for a in actions:
        s = s.get_next_state(a)
    print(s)

    start_node = GameNode(s)
    depth = 3

    print('Heuristic:', max_consecutive_squares_eval(start_node.state))

    start = time.time()
    value, actions = minimax(start_node, depth=3, player='max', heuristic=max_consecutive_squares_eval)
    end = time.time()
    print(f'Minimax: {value}, time = {end - start} sec, principal variation: {actions}')

    start = time.time()
    value, actions = alphabeta(start_node, depth=3, player='max', heuristic=max_consecutive_squares_eval)
    end = time.time()
    print(f'Alpha-beta: {value}, time = {end - start} sec, principal variation: {actions}')
